import streamlit as st
import pandas as pd
import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from llm_config import get_model
import os

# Defining embedding model for vectorization
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

st.title("🔍 DevOps Error Resolution Assistant")

# Function to load csv file
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # Removing extra spaces in column names
    # Creating a new column named 'Error Message' that combines all other columns to store them as embedings
    df['Error Message'] = df[['Error Type', 'Description', 'Possible Causes', 'Solution']].astype(str).agg('-'.join, axis=1)
    return df

# Function to create Normalized Embeddings
def create_embeddings(df):
    vectors = np.array(embedding_model.embed_documents(df['Error Message'].tolist()), dtype='float32')
    # Normalizing the embeddings
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # Vectors are normalized to compare the vectors using Cosine Similarity
    return vectors

# Function to store embedings in FAISS database using cosine similarity
def store_embeddings_faiss(df, vectors, faiss_index_path):
    vector_dimension = vectors.shape[1]  # Getting the dimension of vector(embedding)
    index = faiss.IndexFlatIP(vector_dimension)  # Using Inner Product for Cosine Similarity
    index.add(vectors)
    faiss.write_index(index, faiss_index_path)

    documents = [Document(page_content=text) for text in df['Error Message'].tolist()] # Converting text into document object
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    docstore = {str(i): document for i, document in enumerate(documents)}

    vectorstore = FAISS(
        embedding_function=embedding_model,          # For query
        index=index,                                 # Actual embeddings
        docstore=InMemoryDocstore(docstore),
        index_to_docstore_id=index_to_docstore_id    # Mapping docstore id to the FAISS index indices
    )
    return vectorstore

# Function to load FAISS Documents
def load_faiss_index(faiss_index_path, df):
    if os.path.exists(faiss_index_path):
        index = faiss.read_index(faiss_index_path)
        expected_dimension = create_embeddings(df).shape[1]
        if index.d != expected_dimension:
            st.warning(f"FAISS index dimension mismatch! Rebuilding index...")
            os.remove(faiss_index_path)
            return None  # Force recreation if dimensions doesn't match
        return index
    return None          # Force recreation if file doesn't exist

# Initializing the LLM (Google Gemma) model
def initialize_llm():
    llm = get_model()
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a DevOps assistant. Use the provided context to help resolve user errors.
        If there is no context. Then provide the solution as per your knowledge.
        If a user asks to correct the script or command then correct it. Check twice before giving the solution and take care that only provide the corrected solution and in the last where it commited the errors and where you made changes to the error.
        
        Context: {context}
        
        User Question: {question}
        
        Provide a clear and concise solution.
        """
    )
    return llm, prompt_template

# Setup Retrieval-based QA System (pipeline) - combines FAISS and LLM model
def setup_retrieval_qa(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=initialize_llm()[0],
        retriever=retriever,
        chain_type="stuff",
    )
    return qa_chain

# Loading csv data and setting up FAISS
csv_path = "./sample.csv"
faiss_index_path = "faiss_index.bin"

df = load_data(csv_path)

index = load_faiss_index(faiss_index_path, df)

if index is None:
    vectors = create_embeddings(df)
    vectorstore = store_embeddings_faiss(df, vectors, faiss_index_path)
else:
    documents = [Document(page_content=text) for text in df['Error Message'].tolist()]
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
    query_vector /= np.linalg.norm(query_vector)  # Normalize query vector
    
    if len(query_vector) == vectorstore.index.d:
        response = qa_chain.run(user_question)
        st.subheader("💡 Suggested Solution:")
        st.write(response)
    else:
        st.error(f"⚠️ Query dimension ({len(query_vector)}) does not match FAISS index ({vectorstore.index.d}).")
