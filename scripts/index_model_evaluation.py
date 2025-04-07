import numpy as np
import json
from tqdm import tqdm
from index_search import search_index

def recall_calc(predictions, answer):
    matched = sum(any(ans in pred for pred in predictions) for ans in answer)
    return matched / len(answer)

def simple_index(question, model, index, metadata, docs, top_k = 10):
    q_embed = np.array([model.encode(question, convert_to_tensor=False)])
    D, I = index.search(q_embed, top_k)
    pred = [metadata[i] for i in I[0]]


    return pred

def reranker_index(question, model, index, metadata, docs, cross_encoder, top_k = 10):
    top_k_files_data = search_index(question, model, index, metadata, docs, cross_encoder, top_k = 10) 
        
    top_k_files = [f['file_path'] for f in top_k_files_data]

    return top_k_files

def recall_from_json(ans_path, get_top_k_files, model, index, docs, metadata):
    with open(ans_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    recalls = []
    for item in tqdm(data):
        question = item["question"]
        relevant_files = item["files"]
        
        top_k_files = get_top_k_files(question, model, index, metadata, docs, top_k = 10)
        
        recall = recall_calc(top_k_files, relevant_files)
        recalls.append(recall)

        print(f"Question: {question}")
        print(f"Recall@10: {recall:.2f}")
        print(f"Answer: {relevant_files}")
        print(f"Prediction: {top_k_files[:10]}\n")

    avg_recall = sum(recalls) / len(recalls)
    print(f"\n Average Recall@10: {avg_recall:.2f}")

    return avg_recall