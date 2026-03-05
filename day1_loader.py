from pypdf import PdfReader
import os

pdf_path = "documents/INTRODUCTION PROJECT.pdf"

if not os.path.exists(pdf_path):
    print("PDF not found! Put your PDF inside documents/ folder.")
    exit()

reader = PdfReader(pdf_path)

text = ""
for page in reader.pages:
    text += page.extract_text()

print("Total characters extracted:", len(text))
print("\nFirst 500 characters:\n")
print(text[:500])