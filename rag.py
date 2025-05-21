import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variable
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Extract text from uploaded file
def extract(file):
    if file.type == 'application/pdf':
        text = ""
        # Fix: Use BytesIO to handle file stream properly
        file_bytes = file.read()
        file.seek(0)  # Reset file pointer for potential reuse
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif file.type == "text/plain":
        # Fix: Handle potential encoding issues
        try:
            return file.read().decode("utf-8")
        except UnicodeDecodeError:
            return file.read().decode("latin-1")  # Fallback encoding
    else:
        return ""

# Split text into overlapping chunks
def chunkzation(text, chunk_size=500, overlap=100):
    # Fix: Handle empty text case
    if not text:
        return []
    
    chunkz = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Fix: Check for out of range indices
        if end > len(text):
            end = len(text)
        chunkz.append(text[start:end])
        start += chunk_size - overlap
        # Stop if we've processed the whole text or getting diminishing chunks
        if start >= len(text) or chunk_size - overlap <= 0:
            break
    return chunkz

# Cosine similarity
def cosine_similarity(a, b):
    # Fix: Handle zero vectors to prevent division by zero
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    a = a / norm_a
    b = b / norm_b
    return np.dot(a, b)

# Streamlit App
st.title("RAG using Sentence Transformers + Gemini (No Chroma)")

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state['model'] = None

uploaded = st.file_uploader("Upload a file (pdf/txt):", type=["pdf", "txt"])

if uploaded:
    st.success(f"Uploaded: {uploaded.name}")
    
    # Fix: Add error handling for text extraction
    try:
        raw_text = extract(uploaded)
        if not raw_text:
            st.error("Failed to extract text from the document.")
        else:
            st.subheader("Preview")
            st.write(raw_text[:500])

            st.subheader("Chunks Created")
            chunks = chunkzation(raw_text)
            st.write(f"Number of chunks: {len(chunks)}")

            # Generate embeddings
            # Fix: Load model only once and cache it
            if st.session_state['model'] is None:
                with st.spinner("Loading embedding model..."):
                    st.session_state['model'] = SentenceTransformer('all-MiniLM-L6-v2')
            
            model = st.session_state['model']
            
            # Fix: Handle potential embedding errors
            try:
                with st.spinner("Generating embeddings..."):
                    embeddings = model.encode(chunks, show_progress_bar=True)
                
                query = st.text_input("Enter your question:")
                if query:
                    with st.spinner("Processing your question..."):
                        query_embedding = model.encode([query])[0]

                        # Compute cosine similarity
                        similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
                        top_k = np.argsort(similarities)[-5:][::-1]  # Top 5 most similar chunks

                        relevant_chunks = [chunks[i] for i in top_k]
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

                        # Fix: Check if API key exists
                        if not api_key:
                            st.error("Google API key not found. Please add it to your .env file.")
                        else:
                            # Fix: Configure API only once
                            genai.configure(api_key=api_key)
                            
                            try:
                                gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
                                response = gemini_model.generate_content(prompt)
                                st.markdown("### Answer")
                                st.write(response.text)
                            except Exception as e:
                                st.error(f"Gemini API failed: {e}")
                                st.info("Check your API key and model name. The model name should match what's available in your Google AI Studio account.")
            except Exception as e:
                st.error(f"Error generating embeddings: {e}")
    except Exception as e:
        st.error(f"Error processing file: {e}")