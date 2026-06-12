import requests
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
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
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])

    source_name = os.path.splitext(json_file)[0]
       
    for i, chunk in enumerate(content['chunks']):
        chunk['source_name'] = source_name
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]       
        chunk_id += 1 
        my_dicts.append(chunk)  
     
# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)

vectors = np.vstack(df["embedding"]).astype("float32")

faiss.normalize_L2(vectors)

dimension = vectors.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(vectors)

faiss.write_index(index, "video_index.faiss")

joblib.dump(df, "embeddings.joblib")