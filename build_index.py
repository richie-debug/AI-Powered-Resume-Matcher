import json
import pickle
import faiss
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# Step 1: Load candidates.json
with open("data/candidates.json", "r") as file:
    candidates = json.load(file)

# Step 2 & 3: Extract and prepare combined text
texts = []
metadata = []

for candidate in candidates:
    job_title = candidate.get("job_title", "")
    skills_list = candidate.get("skills", [])
    summary = candidate.get("summary", "")
    
    skills_str = ", ".join(skills_list)
    combined_text = f"Job Title: {job_title}. Skills: {skills_str}. Summary: {summary}"
    
    texts.append(combined_text)
    metadata.append(candidate)

# Step 4: Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, convert_to_numpy=True)

# Step 5: Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Step 6: Save index and metadata
os.makedirs("embeddings", exist_ok=True)
faiss.write_index(index, "embeddings/index.faiss")

with open("embeddings/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ Embedding index and metadata saved successfully.")
