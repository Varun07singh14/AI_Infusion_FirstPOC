import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI


load_dotenv()                 # loads everything from .env

client = OpenAI()

# Page configuration
st.set_page_config(
    page_title="TalkPDF Dashboard",
    page_icon="📄",
    layout="wide"
)

# Title and Subtitle
st.title("💬 TalkPDF Dashboard")
st.caption("Chat with your documents using LangChain, OpenAI, and Chroma DB")

# Sidebar - API Key and File Upload
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API Key")
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
    
    st.divider()
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# Session State Initialization for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Document Processing Pipeline
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    os.remove(tmp_path)
    return vector_store

# Process uploaded file
if uploaded_file and os.environ.get("OPENAI_API_KEY"):
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("Processing PDF document and building vector store..."):
            st.session_state.vector_store = process_pdf(uploaded_file)
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []
            st.success("PDF processed successfully!")

# RAG QA Chain Creation
def get_qa_chain(vector_store):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert assistant for document analysis. "
            "Answer questions strictly using only the provided context. "
            "If the answer is not contained in the context, state that clearly.\n\n"
            "Context:\n{context}"
        )),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📌 Source References"):
                for src in message["sources"]:
                    st.write(f"- **Page {src['page']}**: _{src['content'][:200]}..._")

# Chat Input & Streamlit Response Loop
if user_prompt := st.chat_input("Ask a question about your uploaded PDF..."):
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please enter your OpenAI API key in the sidebar first.")
    elif not st.session_state.vector_store:
        st.error("Please upload a PDF document first.")
    else:
        # User message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching document & generating answer..."):
                qa_chain = get_qa_chain(st.session_state.vector_store)
                response = qa_chain.invoke({"input": user_prompt})
                
                answer = response["answer"]
                sources = [
                    {"page": doc.metadata.get("page", 0) + 1, "content": doc.page_content}
                    for doc in response["context"]
                ]
                
                st.markdown(answer)
                with st.expander("📌 Source References"):
                    for src in sources:
                        st.write(f"- **Page {src['page']}**: _{src['content'][:200]}..._")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })