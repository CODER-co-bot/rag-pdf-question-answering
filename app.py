import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

st.title("📄 Chat with Your PDF (Local RAG)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)

    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Chunking
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    st.success(f"PDF loaded with {len(chunks)} chunks.")

    # Load embedding model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    # FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Load small local LLM
    qa_pipeline = pipeline(
        "text-generation",
        model="google/flan-t5-small"
    )

    query = st.text_input("Ask a question about your PDF:")

    if query:
        query_embedding = embed_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        k = 3
        distances, indices = index.search(query_embedding, k)

        context = "\n".join([chunks[i] for i in indices[0]])

        prompt = f"""
        Answer the question based only on the context below.

        Context:
        {context}

        Question:
        {query}
        """

        result = qa_pipeline(prompt, max_length=256)

        st.subheader("Answer:")
        st.write(result[0]["generated_text"])