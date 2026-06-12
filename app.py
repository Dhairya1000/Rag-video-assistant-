import streamlit as st
import joblib
import numpy as np
import requests
import faiss

# =====================================
# MUST BE FIRST STREAMLIT COMMAND
# =====================================

st.set_page_config(
    page_title="Video RAG Assistant",
    page_icon="🎥",
    layout="wide"
)

# =====================================
# Helpers
# =====================================

def sec_to_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def create_embedding(text):

    try:
        r = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": "nomic-embed-text",
                "input": [text]
            },
            timeout=60
        )

        r.raise_for_status()

        return r.json()["embeddings"][0]

    except Exception as e:
        st.error(f"Embedding Error: {e}")
        return None


def ask_llm(prompt):

    try:

        payload = {
            "model": "gemma2:2b",
            "prompt": prompt,
            "stream": False
        }

        r = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120
        )

        r.raise_for_status()

        return r.json()["response"]

    except Exception as e:
        return f"LLM Error: {e}"


# =====================================
# Load Resources
# =====================================

@st.cache_resource
def load_resources():

    try:

        df = joblib.load("embeddings.joblib")

        index = faiss.read_index("video_index.faiss")

        return df, index

    except Exception as e:

        st.error(f"Error Loading Resources: {e}")
        return None, None


df, index = load_resources()

if df is None or index is None:
    st.stop()

# =====================================
# UI
# =====================================

st.title("🎥 Video RAG Assistant")

st.markdown(
    """
Ask questions from your indexed videos.

The assistant retrieves relevant video chunks using FAISS
and answers using Gemma via Ollama.
"""
)

# =====================================
# Chat History
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================
# Question Input
# =====================================

question = st.chat_input(
    "Ask a question about your videos..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Searching videos..."):

        question_embedding = create_embedding(question)

        if question_embedding is None:
            st.stop()

        query_vector = np.array(
            [question_embedding],
            dtype="float32"
        )

        faiss.normalize_L2(query_vector)

        scores, indices = index.search(
            query_vector,
            k=5
        )

        # =====================================
        # Similarity Check
        # =====================================

        best_score = float(scores[0][0])

        if best_score < 0.25:

            answer = (
                "The provided context does not contain enough information."
            )

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            st.stop()

        # =====================================
        # Safe Retrieval
        # =====================================

        valid_indices = [
            idx for idx in indices[0]
            if idx >= 0
        ]

        retrieved_chunks = df.iloc[valid_indices]

        retrieved_chunks = retrieved_chunks.drop_duplicates(
            subset=["text"]
        )

        context = ""

        sources = []

        for _, row in retrieved_chunks.iterrows():

            start_time = sec_to_time(row["start"])
            end_time = sec_to_time(row["end"])

            context += f"""
Source: {row['source_name']}
Timestamp: {start_time} - {end_time}
Content: {row['text']}
"""

            sources.append(
                f"{row['source_name']} ({start_time} - {end_time})"
            )

        prompt = f"""
You are an AI Video Assistant.

Your job is to answer questions strictly using the retrieved video context.

Instructions:
- Use ONLY the information present in the context.
- Never make up facts.
- If the answer cannot be found in the context, respond:
"The provided context does not contain enough information."
- Mention the source video name.
- Mention the exact timestamp where the answer was found.
- Format timestamps as MM:SS - MM:SS.
- Keep answers concise and factual.

Context:
{context}

Question:
{question}

Answer:
"""

        answer = ask_llm(prompt)

    # =====================================
    # Assistant Message
    # =====================================

    with st.chat_message("assistant"):

        st.markdown(answer)

        with st.expander("📚 Retrieved Sources"):

            for source in sources:
                st.write("•", source)

        st.caption(
            f"Top Similarity Score: {best_score:.3f}"
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )