import numpy as np

def embed_question(question, model):
    return np.array([model.encode(question, convert_to_tensor=False)])

def get_filename_from_chunk_metadata(meta_entry):
    return meta_entry.split("::")[0]

def generate_summary(generator, question, retrieved_docs):
    context = "\n\n".join([
        f"File: {doc['file_path']}\n{doc['code_snippet']}" for doc in retrieved_docs
    ])
    results = []
    prompt = f"""How do these files answer the query? Short summary.
QUERY:
{question}
FILES:
{context}
"""

    result = generator(prompt, max_length=256, do_sample=False)

    
    return result[0]['generated_text'].strip()

def search_index(query, model, index, metadata, docs, cross_encoder, top_k=10, mean_chunk_number=5, sum_generation=False, generator=None):
    query_embedding = embed_question(query, model) # embed the query

    D, I = index.search(query_embedding, top_k * mean_chunk_number) # select chunks
    seen_files = {}
    results = []

    for idx in I[0]: # get filenames from chunks
        file_path = get_filename_from_chunk_metadata(metadata[idx])
        if file_path not in seen_files:
            seen_files[file_path] = docs[idx]

    candidates = [{"file_path": fp, "code_snippet": snippet} for fp, snippet in seen_files.items()] # now we work with files, not chunks

    cross_encoder_input = [(query, cand["code_snippet"]) for cand in candidates]
    scores = cross_encoder.predict(cross_encoder_input) # files evaluation

    for cand, score in zip(candidates, scores):
        cand["score"] = score

    ranked_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True) # sort files in descending order

    if sum_generation:
        return ranked_candidates[:top_k], generate_summary(generator, query, ranked_candidates[:top_k])
    else:
        return ranked_candidates[:top_k]