import pandas as pd
import joblib
import numpy as np
import requests
import faiss

def sec_to_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": text_list
        }
    )

    embedding = r.json()["embeddings"]
    return embedding


# Load embeddings and FAISS index
df = joblib.load("embeddings.joblib")
index = faiss.read_index("video_index.faiss")

# User question
question = input("Ask any question --> ")

# Create embedding for query
question_embedding = create_embedding([question])[0]

query_vector = np.array(
    [question_embedding],
    dtype="float32"
)

faiss.normalize_L2(query_vector)

# Search
scores, indices = index.search(
    query_vector,
    k=5
)

new_df = df.iloc[indices[0]]

# Build context
context = ""

for _, row in new_df.iterrows():

    start_time = sec_to_time(row["start"])
    end_time = sec_to_time(row["end"])

    context += f"""
Source: {row['source_name']}
Timestamp: {start_time} - {end_time}
Content: {row['text']}

"""

# Prompt
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
- If multiple video chunks support the answer, cite all relevant sources.
- Do not mention chunk IDs, embeddings, JSON fields, or internal metadata.

Context:
{context}

Question:
{question}

Answer:
"""

# Save prompt for debugging
with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

def get_ollama_response(prompt_text):

    print("🤖 Thinking... (Asking Gemma)")

    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "gemma2:2b",
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")

    except Exception as e:
        return f"Error connecting to Ollama: {e}"

# Generate answer
print(prompt)

final_answer = get_ollama_response(prompt)

# Add retrieved sources
final_answer += "\n\nRetrieved Sources:\n"

for _, row in new_df.iterrows():
    final_answer += (
        f"\n- {row['source_name']} "
        f"({sec_to_time(row['start'])} - "
        f"{sec_to_time(row['end'])})"
    )

# Save response
with open("response.txt", "w", encoding="utf-8") as f:
    f.write(final_answer)

print("\n" + "=" * 50)
print(final_answer)
print("=" * 50)

print("✅ Answer saved to response.txt")