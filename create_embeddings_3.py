import requests
import os
import json
import pandas as pd
import numpy as np
import joblib 
import faiss

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "nomic-embed-text",
        "input": text_list
    })
    response = r.json()
    if "error" in response:
        print("Embedding model error:", response["error"])
        return []
    return response.get("embeddings", [])

jsons = os.listdir("jsons")  # List all the jsons
my_dicts = []

if os.path.exists("embeddings.joblib") and os.path.exists("video_index.faiss"):
    old_df = joblib.load("embeddings.joblib")
    processed_videos = set(old_df["source_name"])
    index = faiss.read_index("video_index.faiss")
    chunk_id = old_df["chunk_id"].max() + 1

else:
    old_df = pd.DataFrame()
    processed_videos = set()
    index = None
    chunk_id = 0

for json_file in jsons:
    source_name = os.path.splitext(json_file)[0]
    if source_name in processed_videos:
        print(f"Skipping {source_name}")
        continue

    with open(f"jsons/{json_file}",encoding="utf-8") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
    if len(embeddings) != len(content['chunks']):
        print(f"Error in {json_file}")
        continue
       
    for i, chunk in enumerate(content['chunks']):
        chunk['source_name'] = source_name
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]       
        chunk_id += 1 
        my_dicts.append(chunk)  
     
# print(my_dicts)

if len(my_dicts) == 0:
    print("No new videos found.")
    exit()

df = pd.DataFrame.from_records(my_dicts)
vectors = np.vstack(df["embedding"]).astype("float32")

faiss.normalize_L2(vectors)
if index is None:
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
index.add(vectors)
faiss.write_index(index, "video_index.faiss")

updated_df = pd.concat(
    [old_df, df],
    ignore_index=True
)

joblib.dump(
    updated_df,
    "embeddings.joblib"
)

print()
print("Embeddings updated successfully")
print(f"New chunks added : {len(df)}")
print(f"Total chunks : {len(updated_df)}")
print(f"Total vectors in FAISS : {index.ntotal}")