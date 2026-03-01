import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

CSV_PATH = "data/Bhagwad_Gita.csv"

def load_data():
    df = pd.read_csv(CSV_PATH)

    texts = []
    for _, row in df.iterrows():
        chapter = row["Chapter"]
        verse = row["Verse"]
        meaning = row["EngMeaning"]

        combined = f"Chapter {chapter}, Verse {verse}. {meaning}"
        texts.append(combined)

    return texts

def create_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.array(embeddings).astype("float32")

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

if __name__ == "__main__":
    print("Loading data...")
    texts = load_data()

    print("Creating embeddings...")
    embeddings = create_embeddings(texts)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Saving index and texts...")
    faiss.write_index(index, "gita.index")

    with open("verses.pkl", "wb") as f:
        pickle.dump(texts, f)

    print("Done.")