import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import subprocess 
import uuid
import tempfile


@st.cache_resource
def load_models():
    index = faiss.read_index("gita.index")

    with open("verses.pkl", "rb") as f:
        texts = pickle.load(f)

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return index, texts, embedding_model


def retrieve(query, index, texts, embedding_model, k=4):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    context = "\n\n".join([texts[i] for i in indices[0]])
    return context[:1000]


def generate_answer(context, question):
    prompt = f"""
You are a calm and compassionate life guide grounded strictly in the Bhagavad Gita.

Use ONLY the verses provided.
Identify the core teaching.
Explain it clearly in modern English.
Apply it practically to the user's situation.
Respond in one well-developed paragraph.
Do not repeat the question.
Do not repeat instructions.
Do not invent teachings outside the verses.

Verses:
{context}

User Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="gemma:2b-instruct",
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.3,
            "num_predict": 120
        }
    )

    return response["message"]["content"]


def speak(text, voice="Daniel"):
    import tempfile
    import subprocess
    import os

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    subprocess.run([
        "say",
        "-v", voice,
        "-o", wav_path,
        "--data-format=LEI16@22050",
        text
    ])

    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    os.remove(wav_path)

    return audio_bytes


st.title("🕉 Bhagavad Gita Life Guidance Assistant")
st.write("Ask any life question and receive guidance grounded in the Bhagavad Gita.")

index, texts, embedding_model = load_models()

question = st.text_input("Ask your question:")

if question:
    with st.spinner("Reflecting on the Gita..."):
        context = retrieve(question, index, texts, embedding_model)
        answer = generate_answer(context, question)

    st.subheader("Relevant Verses")
    st.write(context)

    st.subheader("Guidance")
    st.write(answer)

    audio_bytes = speak(answer, voice="Daniel")
    st.audio(audio_bytes, format="audio/wav")