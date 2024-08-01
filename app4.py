import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

import os

folder_path = "db"
pdf_folder_path = "pdf"

# Initialize the LLM and other components
cached_llm = Ollama(model="llama3.1")

# Initialize HuggingFaceEmbeddings with appropriate parameters
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

# Streamlit GUI
st.title("AI Document Assistant")

def ai_response(query):
    response = cached_llm.invoke(query)
    return response

def ask_pdf_response(query):
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
    
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    return {"answer": result["answer"], "sources": sources}

def upload_pdfs(files):
    os.makedirs(pdf_folder_path, exist_ok=True)
    docs = []

    for file in files:
        file_name = file.name
        save_file = os.path.join(pdf_folder_path, file_name)
        with open(save_file, "wb") as f:
            f.write(file.getbuffer())

        loader = PDFPlumberLoader(save_file)
        docs.extend(loader.load_and_split())

    chunks = text_splitter.split_documents(docs)

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    vector_store.persist()

    return {"status": "Successfully Uploaded", "total_files": len(files), "total_docs": len(docs), "chunks": len(chunks)}

# Sidebar for file upload
st.sidebar.header("Upload PDF Files")
uploaded_files = st.sidebar.file_uploader("Select PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    response = upload_pdfs(uploaded_files)
    st.sidebar.write(response)

# Main area for querying
query = st.text_input("Enter your query")

if st.button("Get AI Response"):
    response = ai_response(query)
    st.write("Response:", response)

if st.button("Ask PDF"):
    response = ask_pdf_response(query)
    st.write("Answer:", response["answer"])
    st.write("Sources:", response["sources"])
