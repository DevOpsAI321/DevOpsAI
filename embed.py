import streamlit as st
import pandas as pd
import faiss
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document
from llm_config import get_model
import os

# ✅ Use a consistent embedding model
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Streamlit App
st.title("🔍 DevOps Error Resolution Assistant")

# Step 1: Load CSV Dataset
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # Remove extra spaces in column names
    return df

# Step 2: Create Embeddings
def create_embeddings(df):
    vectors = embedding_model.embed_documents(df['Error Message'].astype(str).tolist())
    return np.array(vectors, dtype='float32')

# Step 3: Store in FAISS Database
def store_embeddings_faiss(df, vectors, faiss_index_path):
    d = vectors.shape[1]  # Dimensionality of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(vectors)
    faiss.write_index(index, faiss_index_path)

    documents = [Document(page_content=text) for text in df['Error Message'].astype(str).tolist()]
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    vectorstore = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)}),
        index_to_docstore_id=index_to_docstore_id
    )
    return vectorstore

# Step 4: Load FAISS Index
def load_faiss_index(faiss_index_path, df):
    if os.path.exists(faiss_index_path):
        index = faiss.read_index(faiss_index_path)
        expected_dim = len(create_embeddings(df)[0])

        if index.d != expected_dim:
            st.warning(f"⚠️ FAISS index dimension mismatch! Expected {expected_dim}, found {index.d}. Rebuilding index...")
            os.remove(faiss_index_path)
            return None  # Force recreation
        return index
    return None  # No index found

# Step 5: Initialize LLM
def initialize_llm():
    llm = get_model()
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a DevOps assistant. Use the provided context to help resolve user errors.
        
        Context: {context}
        
        User Question: {question}
        
        Provide a clear and concise solution.
        """
    )
    return llm, prompt_template

# Step 6: Retrieval-based QA System
def setup_retrieval_qa(vectorstore):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=initialize_llm()[0],
        retriever=retriever,
        chain_type="stuff",
    )
    return qa_chain

# Load Data and Setup FAISS
csv_path = "./reduced_to_ten.csv"
faiss_index_path = "faiss_index.bin"
df = load_data(csv_path)

index = load_faiss_index(faiss_index_path, df)
if index is None:
    vectors = create_embeddings(df)
    vectorstore = store_embeddings_faiss(df, vectors, faiss_index_path)
else:
    documents = [Document(page_content=text) for text in df['Error Message'].astype(str).tolist()]
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}
    vectorstore = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)}),
        index_to_docstore_id=index_to_docstore_id
    )

qa_chain = setup_retrieval_qa(vectorstore)

# Streamlit UI for user input
user_question = st.text_input("❓ Enter your error message:")
if user_question:
    query_vector = embedding_model.embed_query(user_question)

    if len(query_vector) == vectorstore.index.d:
        response = qa_chain.run(user_question)
        st.subheader("💡 Suggested Solution:")
        st.write(response)
    else:
        st.error(f"⚠️ Query dimension ({len(query_vector)}) does not match FAISS index ({vectorstore.index.d}).")
