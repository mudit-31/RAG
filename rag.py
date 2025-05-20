import streamlit as st
import fitz  
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


def extract(file):
    if file.type == 'application/pdf':
        text = ""
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    else:
        return ""

def chunkzation(text, chunk_size=500, overlap=100):
    chunkz = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunkz.append(text[start:end])
        start += chunk_size - overlap
    return chunkz


st.title("RAG using Sentence Transformer + Gemini")

uploaded = st.file_uploader("Upload a file (pdf/txt): ", type=["pdf", "txt"])

if uploaded:
    st.success(f"Uploaded: {uploaded.name}")
    raw_text = extract(uploaded)

    st.subheader("Preview")
    st.write(raw_text[:500])

    st.subheader("Chunks Created")
    chunks = chunkzation(raw_text)
    st.write(f"Number of chunks: {len(chunks)}")

 
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, show_progress_bar=True)
    embedding_array = np.array(embeddings).astype("float32")


    index = faiss.IndexFlatL2(embedding_array.shape[1])
    index.add(embedding_array)
    chunk_id_map = {i: chunk for i, chunk in enumerate(chunks)}


    query = st.text_input("Enter your question:")
    if query:
        query_embedding = model.encode([query])[0]
        D, I = index.search(np.array([query_embedding]).astype("float32"), k=5)
        relevant_chunks = [chunk_id_map[i] for i in I[0]]
        context = "\n".join(relevant_chunks)


        prompt = f"""
You are a helpful assistant. Answer the following question based only on the provided context. 
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{query}

Answer:
"""


        genai.configure(api_key=api_key) 
        gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

        try:
            response = gemini_model.generate_content(prompt)
            st.markdown("### Answer")
            st.write(response.text)
        except Exception as e:
            st.error(f"Error: {e}")
