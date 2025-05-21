import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# Initialize Streamlit configuration
st.set_page_config(page_title="RAG System", layout="wide")

# API Key Management - With better path handling for .env files
def setup_api_key():
    """
    Sets up the API key with multiple fallback mechanisms.
    Handles .env files correctly even when they're in .gitignore.
    """
    # Try loading from different possible .env locations
    possible_env_paths = [
        Path('.env'),                    # Current directory
        Path(os.getcwd()) / '.env',      # Explicit current working directory
        Path(__file__).parent / '.env',  # Directory where this script is located
    ]
    
    api_key = None
    
    # Try each possible path
    for env_path in possible_env_paths:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                st.sidebar.success(f"Loaded API key from {env_path}")
                break
    
    # If not found in .env files, try direct environment variable
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
        
    # If still not found, check Streamlit secrets
    if not api_key and hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
        api_key = st.secrets['GOOGLE_API_KEY']
    
    # If still not found or retrieved from session state if previously entered
    if not api_key and 'api_key' in st.session_state:
        api_key = st.session_state['api_key']
    
    # Finally, allow manual entry
    if not api_key:
        st.sidebar.warning("API key not found in environment or secrets.")
        with st.sidebar:
            st.markdown("### Enter API Key")
            entered_key = st.text_input("Google API Key:", type="password", key="manual_api_key")
            
            if entered_key and entered_key.strip():
                if st.button("Save API Key"):
                    api_key = entered_key.strip()
                    st.session_state['api_key'] = api_key
                    st.success("API key saved for this session!")
                    st.experimental_rerun()
    
    return api_key

# Extract text from uploaded file
def extract(file):
    if file.type == 'application/pdf':
        text = ""
        file_bytes = file.read()
        file.seek(0)  # Reset file pointer
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif file.type == "text/plain":
        try:
            return file.read().decode("utf-8")
        except UnicodeDecodeError:
            file.seek(0)  # Reset file pointer
            return file.read().decode("latin-1")  # Fallback encoding
    else:
        return ""

# Split text into overlapping chunks
def chunkzation(text, chunk_size=500, overlap=100):
    if not text:
        return []
    
    chunkz = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        chunkz.append(text[start:end])
        start += chunk_size - overlap
        if start >= len(text):
            break
    return chunkz

# Cosine similarity
def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    a = a / norm_a
    b = b / norm_b
    return np.dot(a, b)

# Main application
def main():
    st.title("RAG using Sentence Transformers + Gemini (No Chroma)")
    
    # Get and configure API key
    api_key = setup_api_key()
    
    # Configure Gemini API if key is available
    if api_key:
        genai.configure(api_key=api_key)
        with st.sidebar:
            st.success("✅ API key configured successfully!")
    else:
        st.error("No Google API key found. Please enter your API key in the sidebar.")
        return  # Exit early if no API key
    
    # Initialize session state for model
    if 'model' not in st.session_state:
        st.session_state['model'] = None
        
    # File upload section
    uploaded = st.file_uploader("Upload a file (pdf/txt):", type=["pdf", "txt"])
    
    if uploaded:
        st.success(f"Uploaded: {uploaded.name}")
        
        try:
            with st.spinner("Extracting text..."):
                raw_text = extract(uploaded)
                
            if not raw_text:
                st.error("Failed to extract text from the document.")
                return
                
            st.subheader("Preview")
            st.write(raw_text[:500])
    
            st.subheader("Chunks Created")
            chunks = chunkzation(raw_text)
            st.write(f"Number of chunks: {len(chunks)}")
    
            # Generate embeddings
            if st.session_state['model'] is None:
                with st.spinner("Loading embedding model..."):
                    st.session_state['model'] = SentenceTransformer('all-MiniLM-L6-v2')
            
            model = st.session_state['model']
            
            with st.spinner("Generating embeddings..."):
                embeddings = model.encode(chunks, show_progress_bar=True)
            
            # Question answering section
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
    
                    try:
                        gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
                        response = gemini_model.generate_content(prompt)
                        st.markdown("### Answer")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Gemini API failed: {e}")
                        
                        # Help diagnose specific API errors
                        error_str = str(e).lower()
                        if "authentication" in error_str or "auth" in error_str or "invalid" in error_str:
                            st.warning("This looks like an API key issue. Please check if your key is valid.")
                        elif "model" in error_str:
                            st.warning("The model name might be incorrect or not available for your account.")
                        elif "quota" in error_str or "limit" in error_str:
                            st.warning("You may have reached your API usage limits.")
                        
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Please try again or check your file format.")

# Run the app
if __name__ == "__main__":
    main()