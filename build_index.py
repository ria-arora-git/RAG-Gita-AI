import os
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle


def extract_text(pdf_path):
    full_text=""

    with pdfplumber.open("The Bhagavad Gita.pdf") as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

    return full_text

def clean_text(full_text):
    lines = full_text.split("\n")
    cleaned_text = []

    for line in lines:
        line = line.strip()

        if line.isdigit() or len(line) == 0:
            continue

        cleaned_text.append(line)
        
    cleaned_lines = " ".join(cleaned_text)
    return cleaned_lines

        
def chunk_text(cleaned_lines):
    pattern = r"\(\d+\.\d+\)"
    splits = re.split(pattern, cleaned_lines)

    chunks = []

    for chunk in splits:
        chunk = chunk.strip()
        if len(chunk)>100:
            chunks.append(chunk)

    return chunks


def create_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    return embeddings

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index

if __name__ == "__main__":
    print("Extracting text...")
    text = extract_text("/Users/riaarora/Desktop/AI PROJECTS RESUME/rag_gita_ai/The Bhagavad Gita.pdf")

    print("Cleaning text...")
    cleaned = clean_text(text)

    print("Chunking text...")
    chunks = chunk_text(cleaned)

    print("Creating embeddings...")
    embeddings = create_embeddings(chunks)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Saving index and chunks...")
    faiss.write_index(index, "gita.index")

    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    print("All done")