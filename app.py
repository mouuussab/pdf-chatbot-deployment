import streamlit as st
import os
import fitz
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
PDF_PATH = "my_module.pdf"
VECTORSTORE_PATH = "faiss_index"
TOP_K_RETRIEVAL = 5
TOP_N_RERANK = 1

# === THE FIX IS HERE ===
# We are going back to the structured ChatPromptTemplate, which is the correct
# way to interact with a "Chat" model like ChatGroq.
system_prompt = """You are a text extraction robot. Your only function is to answer the user's question using ONLY the information found in the CONTEXT provided.
- Do not use any external knowledge or make assumptions.
- If the CONTEXT contains a list that answers the question, reproduce that list exactly.
- If the CONTEXT does not contain the answer, you must state only: "The document does not contain the answer to this question."
"""
human_prompt = "CONTEXT:\n{context}\n\nQUESTION:\n{question}"

# --- Core Functions ---
def create_vectorstore_if_needed():
    if not os.path.exists(VECTORSTORE_PATH):
        with st.spinner("First-time setup: Processing PDF..."):
            doc = fitz.open(PDF_PATH)
            raw_text = "".join([page.get_text() for page in doc])
            doc.close()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
            chunks = text_splitter.split_text(raw_text)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
            vectorstore.save_local(VECTORSTORE_PATH)
        st.toast("Vector store created successfully!")

@st.cache_resource
def load_resources(groq_api_key):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
    return retriever, reranker_model, llm

# --- Main App Logic ---
def main():
    st.set_page_config(page_title="PDF AI Assistant", page_icon="🚀")
    st.title("AI Assistant for the PM Module")
    st.info("Powered by Groq Llama 3 for lightning-fast answers.")

    if "GROQ_API_KEY" not in st.secrets or not st.secrets["GROQ_API_KEY"]:
        st.error("GROQ_API_KEY not found in Streamlit Secrets. Please add it.")
        st.stop()

    create_vectorstore_if_needed()

    try:
        api_key = st.secrets["GROQ_API_KEY"]
        retriever, reranker_model, llm = load_resources(groq_api_key=api_key)
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        return

    # === AND THE FIX IS HERE ===
    # We construct the ChatPromptTemplate from our system and human prompts.
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])
    
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
            with st.spinner("Searching document and generating answer..."):
                initial_docs = retriever.invoke(user_question)
                if not initial_docs:
                    st.write("I couldn't find any relevant information.")
                    st.session_state.messages.append({"role": "assistant", "content": "I couldn't find any relevant information."})
                    st.stop()

                query_doc_pairs = [[user_question, doc.page_content] for doc in initial_docs]
                scores = reranker_model.score(query_doc_pairs)
                reranked_results = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
                
                final_doc = reranked_results[0][1]
                context_text = final_doc.page_content

                response = rag_chain.stream({"context": context_text, "question": user_question})
                generative_answer = st.write_stream(response)
                
                with st.expander("Show Source Used"):
                    st.write(context_text)

        st.session_state.messages.append({"role": "assistant", "content": generative_answer})

if __name__ == '__main__':
    main()
