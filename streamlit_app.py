import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from gtts import gTTS
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
    return context[:2000]


def generate_answer(context, question):
    prompt = f"""
You are a wise and compassionate spiritual guide grounded in the Bhagavad Gita.

Use ONLY the verses below to answer the question.

Explain the spiritual teaching clearly in modern English.
Apply it thoughtfully to the user's situation.
Write one calm, meaningful paragraph.

Verses:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="phi3:mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


def speak(text):
    tts = gTTS(text)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)

    return temp_file.name


st.title("🕉 Bhagavad Gita Life Guidance Assistant")
st.write("Ask any life question and receive guidance grounded in the Bhagavad Gita.")

index, texts, embedding_model = load_models()

question = st.text_input("Ask your question:")

if question:
    with st.spinner("Reflecting on the Gita..."):
        context = retrieve(question, index, texts, embedding_model)
        answer = generate_answer(context, question)

    st.subheader("Retrieved Verses")
    st.write(context)

    st.subheader("Guidance")
    st.write(answer)

    audio_path = speak(answer)
    st.audio(audio_path)