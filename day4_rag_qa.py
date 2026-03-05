from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# ----------------------------
# 1. Load PDF
# ----------------------------
pdf_path = "documents/INTRODUCTION PROJECT.pdf"
reader = PdfReader(pdf_path)

text = ""
for page in reader.pages:
    text += page.extract_text()

# ----------------------------
# 2. Chunk text
# ----------------------------
chunk_size = 500
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ----------------------------
# 3. Create embeddings
# ----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("Vector database ready.")

# ----------------------------
# 4. Load Local LLM
# ----------------------------
qa_pipeline = pipeline(
    "text-generation",
    model="google/flan-t5-small"
)
print("Local LLM loaded.")

# ----------------------------
# 5. RAG Loop
# ----------------------------
while True:
    query = input("\nAsk a question (type 'quit' to exit): ")

    if query.lower() == "quit":
        break

    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    k = 3
    distances, indices = index.search(query_embedding, k)

    context = "\n".join([chunks[i] for i in indices[0]])

    prompt = f"""
    Answer the question based on the context below.

    Context:
    {context}

    Question:
    {query}
    """

    result = qa_pipeline(prompt, max_length=256)

    print("\nAnswer:\n", result[0]["generated_text"])