#import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()                 # loads everything from .env

client = OpenAI()


def process_pdf_and_answer(pdf_file, query):
    if pdf_file is None:
        return "Please upload a PDF file first."

    if not query.strip():
        return "Please enter a question."

    try:
        # 1. Load the PDF file
        loader = PyPDFLoader(pdf_file.name)
        docs = loader.load()

        # 2. Split text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        # 3. Create embeddings and store in Chroma
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )
        retriever = vectorstore.as_retriever()

        # 4. Set up LLM and RAG prompt chain
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know.\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(
            llm, prompt
        )

        rag_chain = create_retrieval_chain(
            retriever, question_answer_chain
        )

        # 5. Invoke the chain
        response = rag_chain.invoke({"input": query})

        return response["answer"]

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Build a clean Gradio Web Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄 PDF QA Chatbot with LangChain & OpenAI")
    gr.Markdown("Upload a PDF document and ask questions about its content instantly.")
    
    with gr.Row():
        file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
    
    with gr.Row():
        question_input = gr.Textbox(label="Ask a Question", placeholder="What is the main topic of this document?")
    
    submit_btn = gr.Button("Get Answer", variant="primary")
    output_box = gr.Textbox(label="AI Response", lines=5)

    submit_btn.click(
        fn=process_pdf_and_answer,
        inputs=[file_input, question_input],
        outputs=output_box
    )

if __name__ == "__main__":
    demo.launch()