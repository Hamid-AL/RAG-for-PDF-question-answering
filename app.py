import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pdfplumber
import os
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_cohere import CohereEmbeddings
import cohere


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    return text


def get_text_chunks(text):
    SEPARATORS = ["\n", ".", ","]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True,
        strip_whitespace=True,
        separators=SEPARATORS,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def handle_userinput(user_question):
    if "vectorstore" in st.session_state and st.session_state.vectorstore:
        retrieved_docs = st.session_state.vectorstore.similarity_search(user_question)
        retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
        context = "\nExtracted documents:\n" + "".join(
            [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
        )

        cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
        response = cohere_client.chat(
            chat_history=[
                {"role": "USER", "message": """Using the information contained in the context,
                give a comprehensive answer to the question.
                Respond only to the question asked, response should be concise and relevant to the question.
                If the answer cannot be deduced from the context, say 'I have no answer'."""},
                {"role": "CHATBOT", "message": "Please provide the question and the context."},
            ],
            message=f"Question:\n{user_question}\Context:{context}\nResponse:\n",
        )
        st.session_state.messages.append({"role": "assistant", "content": response.text})
    else:
        st.warning("Please process the PDF documents first.")


def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with multiple PDFs', page_icon=':books:')
    

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.header('Chat with multiple PDFs :books:')
    user_question = st.chat_input("Ask a question about your documents:")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader('Your Documents')
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type="pdf")
        
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vectorstore = get_vectorstore(text_chunks)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


if __name__ == '__main__':
    main()
