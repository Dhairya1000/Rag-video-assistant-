import streamlit as st
import joblib
import numpy as np
import requests
import faiss
import os
import subprocess

# =====================================
# MUST BE FIRST STREAMLIT COMMAND
# =====================================

st.set_page_config(
    page_title="Video RAG Assistant",
    page_icon="🎥",
    layout="wide"
)

# =====================================
# Create folders if not exist
# =====================================

os.makedirs("videos", exist_ok=True)

# =====================================
# Helpers
# =====================================

def sec_to_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)

    return f"{minutes:02d}:{seconds:02d}"


def create_embedding(text):

    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": [text]
        }
    )

    r.raise_for_status()

    return r.json()["embeddings"][0]


def ask_llm(prompt):

    payload = {

        "model": "gemma2:2b",

        "prompt": prompt,

        "stream": False

    }

    r = requests.post(

        "http://localhost:11434/api/generate",

        json=payload

    )

    r.raise_for_status()

    return r.json()["response"]


# =====================================
# Load Resources
# =====================================

@st.cache_resource
def load_resources():

    if not os.path.exists("embeddings.joblib"):

        return None, None

    if not os.path.exists("video_index.faiss"):

        return None, None

    df = joblib.load("embeddings.joblib")

    index = faiss.read_index("video_index.faiss")

    return df, index


df, index = load_resources()

# =====================================
# Sidebar
# =====================================

with st.sidebar:

    st.header("📂 Videos")

    videos = [

        f for f in os.listdir("videos")

        if f.endswith(

            (".mp4", ".mkv", ".avi", ".mov")

        )

    ]

    if videos:

        for video in videos:

            st.markdown(f"📹 **{video}**")

    else:

        st.info("No videos found.")

    st.divider()

    uploaded_video = st.file_uploader(

        "Upload New Video",

        type=["mp4", "mkv", "avi", "mov"]

    )

    if uploaded_video:

        save_path = os.path.join(

            "videos",

            uploaded_video.name

        )

        if os.path.exists(save_path):

            st.warning("⚠️ Video already exists!")

        else:

            with open(save_path, "wb") as f:

                f.write(

                    uploaded_video.getbuffer()

                )

            st.success(

                f"✅ {uploaded_video.name} uploaded!"

            )

            st.info(

                "Run your processing scripts to index it."

            )

if st.button("🚀 Process New Videos"):

    with st.spinner("Processing videos..."):

        p1 = subprocess.run(
            ["python", "process_video_1.py"],
            capture_output=True,
            text=True
        )

        st.text(p1.stdout)

        if p1.returncode != 0:

            st.error("Error in process_video_1.py")

            st.text(p1.stderr)

            st.stop()


        p2 = subprocess.run(
            ["python", "create_chunk_of_audios_2.py"],
            capture_output=True,
            text=True
        )

        st.text(p2.stdout)

        if p2.returncode != 0:

            st.error("Error in create_chunk_of_audios_2.py")

            st.text(p2.stderr)

            st.stop()


        p3 = subprocess.run(
            ["python", "create_embeddings_3.py"],
            capture_output=True,
            text=True
        )

        st.text(p3.stdout)

        if p3.returncode != 0:

            st.error("Error in create_embeddings_3.py")

            st.text(p3.stderr)

            st.stop()

    st.success("✅ New videos processed!")

    st.cache_resource.clear()

    st.rerun()

# =====================================
# Main UI
# =====================================

st.title("🎥 Video RAG Assistant")

st.write(

    "Ask questions from your indexed videos."

)

# =====================================
# Chat History
# =====================================

if "messages" not in st.session_state:

    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# =====================================
# User Question
# =====================================

question = st.chat_input(

    "Ask something about your videos..."

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

        question_embedding = create_embedding(

            question

        )

        query_vector = np.array(

            [question_embedding],

            dtype="float32"

        )

        faiss.normalize_L2(query_vector)

        scores, indices = index.search(

            query_vector,

            k=5

        )

        valid_indices = [

            i for i in indices[0]

            if i >= 0

        ]

        retrieved_chunks = df.iloc[valid_indices]

        context = ""

        sources = []

        for _, row in retrieved_chunks.iterrows():

            start_time = sec_to_time(

                row["start"]

            )

            end_time = sec_to_time(

                row["end"]

            )

            context += f"""
Source: {row['source_name']}
Timestamp: {start_time} - {end_time}
Content: {row['text']}
"""

            sources.append({

    "source_name": row["source_name"],

    "start": start_time,

    "end": end_time,

    "text": row["text"]

})

        prompt = f"""
You are an AI Video Assistant.

Answer ONLY from the provided context.

Rules:
- Use ONLY the information present in the context.
- Never make up facts.
- If the answer cannot be found in the context, respond:
"The provided context does not contain enough information."
- Mention the source video name.
- Mention the exact timestamp.
- Format timestamps as MM:SS - MM:SS.

Context:
{context}

Question:
{question}

Answer:
"""

        answer = ask_llm(prompt)

    with st.chat_message("assistant"):

        st.markdown(answer)

        with st.expander(
    "📚 Retrieved Chunks",
    expanded=True
):
            for src in sources:

                st.markdown(
            f"""
### 📹 {src['source_name']}

**⏱ Timestamp:** {src['start']} - {src['end']}

**📝 Retrieved Content:**

{src['text']}
"""
        )

        st.divider()

    st.session_state.messages.append(

        {

            "role": "assistant",

            "content": answer

        }

    )