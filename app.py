import os
import streamlit as st
import pickle
import tempfile
import google.generativeai as palm  
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

st.title('Multi-PDF Q&A System 📚🤖')
st.write('Upload multiple PDFs for a subject and ask questions. The chatbot will search all PDFs for answers.')

st.sidebar.title('Upload Subject PDFs')
uploaded_files = st.sidebar.file_uploader("Choose multiple PDF files", type="pdf", accept_multiple_files=True)
process_pdf_clicked = st.sidebar.button('Process PDFs')

main_placeholder = st.empty()

if process_pdf_clicked and uploaded_files:
    all_docs = []
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.getvalue())
            temp_pdf_path = temp_pdf.name 

        loader = UnstructuredPDFLoader(temp_pdf_path)
        data = loader.load()
        main_placeholder.text(f'Processing {uploaded_file.name}... ✅')

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(data)

        all_docs.extend(docs)  # Collect all documents from all PDFs

        os.remove(temp_pdf_path)  # Clean up temporary file

    # ✅ Convert all PDFs into a single FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(all_docs, embeddings)

    with open('vector_db.pkl', 'wb') as f:
        pickle.dump(vector_db, f)

    st.sidebar.success('✅ All PDFs processed successfully!')

query = main_placeholder.text_input('Ask a question:')

if query:
    try:
        with open('vector_db.pkl', 'rb') as f:
            vector_db = pickle.load(f)

        palm_api_key = os.getenv("GOOGLE_PALM_API_KEY")
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=palm_api_key, temperature=0)

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an expert in chemistry Teacher. Use the provided textbook data to answer questions.\n\n"
                "**Context:**\n{context}\n\n"
                "**Question:** {question}\n\n"
                "Provide a clear and precise answer in a structured manner."
            ),
        )

        chain = RetrievalQA.from_llm(llm=llm, retriever=vector_db.as_retriever(), prompt=prompt_template)

        response = chain.invoke({"query": query})
        answer = response["result"]

        st.write(f'**Answer:** {answer}')
    
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
