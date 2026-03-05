from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import os

# Load PDF
pdf_path = "documents/INTRODUCTION PROJECT.pdf"

reader = PdfReader(pdf_path)

text = ""
for page in reader.pages:
    text += page.extract_text()

print("Total characters:", len(text))

# ----------------------------
# 1. Split into chunks
# ----------------------------
chunk_size = 500
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

print("Total chunks created:", len(chunks))

# ----------------------------
# 2. Load embedding model
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# 3. Convert chunks to embeddings
# ----------------------------
embeddings = model.encode(chunks)

print("Embedding shape:", embeddings.shape)