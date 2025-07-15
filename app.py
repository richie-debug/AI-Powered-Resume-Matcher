import streamlit as st
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ----------------- CONFIG ------------------
genai.configure(api_key="YOUR_GEMINI_KEY")
model = SentenceTransformer("all-MiniLM-L6-v2")
gemini_model = genai.GenerativeModel("gemini-2.5-pro")

# ----------------- FUNCTIONS ------------------
@st.cache_resource
def load_faiss_and_metadata():
    index = faiss.read_index("embeddings/index.faiss")
    with open("embeddings/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

def generate_explanation(job_description, candidate):
    name = candidate.get("name", "")
    job_title = candidate.get("job_title", "")
    skills = ", ".join(candidate.get("skills", []))
    summary = candidate.get("summary", "")

    prompt = f"""
You are an AI assistant for recruiters.

Here is a job description:
{job_description}

Here is a candidate:
Name: {name}
Job Title: {job_title}
Skills: {skills}
Summary: {summary}

Explain why this candidate is a good fit for the job. Keep it short and clear.
"""
    response = gemini_model.generate_content(prompt)
    return response.text

def match_candidates(job_description, top_n=5):
    query_vec = model.encode([job_description], convert_to_numpy=True)
    query_vec = normalize(query_vec)

    faiss_index, metadata = load_faiss_and_metadata()
    candidate_vectors = faiss_index.reconstruct_n(0, faiss_index.ntotal)
    candidate_vectors = normalize(candidate_vectors)

    cosine_index = faiss.IndexFlatIP(candidate_vectors.shape[1])
    cosine_index.add(candidate_vectors)

    scores, indices = cosine_index.search(query_vec, top_n)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        candidate = metadata[idx]
        candidate["score"] = round(float(score), 4)
        candidate["explanation"] = generate_explanation(job_description, candidate)
        results.append(candidate)
    return results

# ----------------- UI ------------------
st.title("🔍 AI-Powered Resume Matcher")

st.markdown("Upload the candidate data and paste a job description to find top matches.")

# Upload candidate JSON file
uploaded_file = st.file_uploader("📄 Upload candidates.json", type="json")

# Load the file if uploaded
if uploaded_file is not None:
    with open("data/candidates.json", "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ Uploaded successfully!")

# Job description input
job_description = st.text_area("📝 Enter Job Description")

# Match button
if st.button("🔎 Match Candidates"):
    if not job_description:
        st.error("Please enter a job description.")
    else:
        with st.spinner("Matching candidates..."):
            matches = match_candidates(job_description, top_n=5)

        # Display matches
        for i, cand in enumerate(matches, 1):
            st.subheader(f"Match #{i}: {cand['name']} ({cand['job_title']})")
            st.write(f"**Score:** {cand['score']}")
            st.write(f"**Skills:** {', '.join(cand['skills'])}")
            st.write(f"**Explanation:** {cand['explanation']}")
            st.markdown("---")
