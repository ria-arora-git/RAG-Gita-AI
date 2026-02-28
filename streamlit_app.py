import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pickle
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import tempfile

torch.set_num_threads(1)


@st.cache_resource
def load_models():
    index = faiss.read_index("gita.index")

    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base"
    )

    device = "cpu"
    model.to(device)

    return index, chunks, embedding_model, tokenizer, model, device


def retrieve(query, index, chunks, embedding_model, k=5):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    context = ""
    for i in indices[0]:
        context += chunks[i] + "\n\n"

    return context


def generate_answer(context, question, tokenizer, model, device):
    short_context = context[:1500]

    prompt = f"""
Answer the question using only the Bhagavad Gita context below.

Context:
{short_context}

Question:
{question}

Provide a clear and meaningful explanation.
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


def speak(answer):
    tts = gTTS(answer)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name


st.title("🕉 Bhagavad Gita AI Assistant")

st.write("Ask any question about the Bhagavad Gita and receive a spiritually grounded explanation.")

index, chunks, embedding_model, tokenizer, model, device = load_models()

question = st.text_input("Ask your question:")

if question:
    with st.spinner("Searching scripture and generating explanation..."):
        context = retrieve(question, index, chunks, embedding_model)
        answer = generate_answer(context, question, tokenizer, model, device)

    st.subheader("Answer")
    st.write(answer)

    audio_path = speak(answer)
    st.audio(audio_path)