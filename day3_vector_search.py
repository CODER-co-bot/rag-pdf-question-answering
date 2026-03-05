from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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

print("Total chunks:", len(chunks))

# ----------------------------
# 3. Create embeddings
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# Convert to numpy float32 (required for FAISS)
embeddings = np.array(embeddings).astype("float32")

# ----------------------------
# 4. Create FAISS index
# ----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index built.")

# ----------------------------
# 5. Search loop
# ----------------------------
while True:
    query = input("\nAsk a question (type 'quit' to exit): ")

    if query.lower() == "quit":
        break

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    k = 3  # number of similar chunks to retrieve
    distances, indices = index.search(query_embedding, k)

    print("\nTop relevant chunks:\n")
    for i in indices[0]:
        print(chunks[i])
        print("-" * 50)