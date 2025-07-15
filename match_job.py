import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index and metadata
index = faiss.read_index("embeddings/index.faiss")
with open("embeddings/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load the same embedding model used during index creation
model = SentenceTransformer("all-MiniLM-L6-v2")

def normalize(vectors):
    """Normalize vectors to use cosine similarity with FAISS (which uses dot product)."""
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

def match_job_description(job_description, top_n=5):
    # Embed the job description
    query_embedding = model.encode([job_description], convert_to_numpy=True)
    query_embedding = normalize(query_embedding)  # Normalize for cosine similarity

    # Normalize candidate vectors too
    index_vectors = index.reconstruct_n(0, index.ntotal)
    index_cosine = faiss.IndexFlatIP(index.d)  # Inner product index ≈ cosine after normalization
    index_cosine.add(normalize(index_vectors))

    # Search top-N matches
    scores, indices = index_cosine.search(query_embedding, top_n)

    # Format results
    results = []
    for idx, score in zip(indices[0], scores[0]):
        candidate = metadata[idx]
        results.append({
            "id": candidate.get("id"),
            "name": candidate.get("name"),
            "job_title": candidate.get("job_title"),
            "skills": candidate.get("skills"),
            "score": round(float(score), 4)
        })

    return results

# Example usage
if __name__ == "__main__":
    job_desc = input("Enter job description:\n")
    top_matches = match_job_description(job_desc, top_n=5)
    for i, match in enumerate(top_matches, start=1):
        print(f"\nTop Match #{i}:")
        print(f"ID: {match['id']}")
        print(f"Name: {match['name']}")
        print(f"Title: {match['job_title']}")
        print(f"Skills: {', '.join(match['skills'])}")
        print(f"Score: {match['score']}")
