import streamlit as st
import os
import re
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

load_dotenv()

# Load the NVIDIA API key
api_key = os.getenv("NVIDIA_API_KEY")
if api_key:
    os.environ['NVIDIA_API_KEY'] = api_key
else:
    st.error("NVIDIA_API_KEY not found in environment variables.")

def clean_text(text):
    # Remove invalid characters
    return re.sub(r'[^\x00-\x7F]+', '', text)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./coffee")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading

        if not st.session_state.docs:
            st.error("No documents loaded.")
            return

        # Clean the text of each document
        for doc in st.session_state.docs:
            doc.page_content = clean_text(doc.page_content)
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting

        if not st.session_state.final_documents:
            st.error("No documents after splitting.")
            return
        
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings
        
        if not st.session_state.vectors:
            st.error("Vector embedding failed.")
            return

# Use CSS to center the title
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Apply the CSS class to the title
st.markdown('<h1 class="centered-title">Art of Coffee</h1>', unsafe_allow_html=True)

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

st.image('coffee.jpg')

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Questions: {input}
    """
)

prompt1 = st.text_input("Iced, hot or coffee cocktail?")

if st.button("Brew me first"):
    vector_embedding()
    st.write("Your coffee is ready")

if prompt1:
    if "vectors" not in st.session_state:
        st.error("Vectors are not initialized. Please click 'Brew me first'.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(f"Response time: {time.process_time() - start:.2f} seconds")
        st.write(response['answer'])

        # With a Streamlit expander
        with st.expander("Similar Coffee Recipes"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("-------------------------------")
