import streamlit as st
import os
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
# These files must be in your GitHub repository
PDF_PATH = "my_module.pdf"
MODEL_PATH = "mistral-7b-instruct-v0.2.Q4_0.gguf" 
VECTORSTORE_PATH = "faiss_index" # This will be created automatically

TOP_K_RETRIEVAL = 5
TOP_N_RERANK = 1

# --- Strict Prompt Template ---
prompt_template = """
### INSTRUCTION:
You are a text extraction robot. Your only function is to answer the user's question using ONLY the information found in the CONTEXT below.
- Do not use any external knowledge.
- If the CONTEXT contains a list that answers the question, reproduce that list exactly.
- If the CONTEXT does not contain the answer, you must state only: "The document does not contain the answer to this question."

### CONTEXT:
{context}

### QUESTION:
{question}

### RESPONSE:
"""

# --- Core Functions ---

# This function combines the logic from your "CREATE THE VECTORSTORE" cells
def create_vectorstore_if_needed():
    """Creates the vector store if it doesn't already exist in the deployment environment."""
    if not os.path.exists(VECTORSTORE_PATH):
        st.write("First-time setup: Creating the vector store from the PDF...")
        with st.spinner("Processing PDF, this may take a moment..."):
            doc = fitz.open(PDF_PATH)
            raw_text = "".join([page.get_text() for page in doc])
            doc.close()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200, chunk_overlap=200, length_function=len
            )
            chunks = text_splitter.split_text(raw_text)
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
            vectorstore.save_local(VECTORSTORE_PATH)
        st.success("Vector store created! The app is now ready.")

# This function is from your "RUN THE APP" cell's script
@st.cache_resource
def load_resources():
    """Loads all necessary models and the vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

    llm = CTransformers(
        model=MODEL_PATH,
        model_type='mistral',
        config={'max_new_tokens': 2048, 'temperature': 0.1, 'context_length': 4096}
        # NOTE: gpu_layers is removed because Streamlit Community Cloud uses CPU.
    )
    return retriever, reranker_model, llm

# --- Main App Logic (from your app_script_content) ---
def main():
    st.set_page_config(page_title="PDF AI Assistant")
    st.title("AI Assistant for the PM Module")
    st.info("This assistant answers questions based on the 'Progiciels et Management' course material.")

    # Run the vector store creation at startup if needed
    create_vectorstore_if_needed()

    try:
        retriever, reranker_model, llm = load_resources()
    except Exception as e:
        st.error(f"Failed to load resources. Are the PDF and model files in the repository? Error: {e}")
        return

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    rag_chain = prompt | llm | StrOutputParser()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("Ask a question based on your document:"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer... (This may be slow on CPU)"):
                initial_docs = retriever.invoke(user_question)
                generative_answer = ""
                if initial_docs:
                    query_doc_pairs = [[user_question, doc.page_content] for doc in initial_docs]
                    scores = reranker_model.score(query_doc_pairs)
                    reranked_results = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
                    
                    final_docs = [doc for score, doc in reranked_results[:TOP_N_RERANK]]
                    context_text = "\\n\\n---\\n\\n".join([doc.page_content for doc in final_docs])

                    response = rag_chain.stream({"context": context_text, "question": user_question})
                    generative_answer = st.write_stream(response)
                    
                    with st.expander("Show Source Used"):
                        st.write(final_docs[0].page_content)
                else:
                    generative_answer = "I couldn't find any relevant information."
                    st.markdown(generative_answer)

        st.session_state.messages.append({"role": "assistant", "content": generative_answer})

if __name__ == '__main__':
    main()
