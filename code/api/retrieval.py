from dotenv import load_dotenv
import os
import json
import numpy as np
import time
from openai import OpenAI
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from api.prompts import initial_query
from beir.datasets.data_loader import GenericDataLoader

# 환경설정
load_dotenv()
DATASET_NAME = "nfcorpus"
EMBEDDING_DIR = os.path.join("datasets", DATASET_NAME, "cached_embeddings")
TOP_K = 5
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Raw corpus 불러오기 (corpus.jsonl 기준)
def load_raw_corpus():
    corpus_path = "datasets/nfcorpus/nfcorpus_raw/corpus.jsonl"
    corpus_dict = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus_dict[doc["_id"]] = doc
    return corpus_dict

# 코퍼스 임베딩 불러오기
def load_corpus_embeddings():
    embeddings = np.load(os.path.join(EMBEDDING_DIR, f"{DATASET_NAME}_corpus_embeddings.npy"))
    with open(os.path.join(EMBEDDING_DIR, f"{DATASET_NAME}_corpus_ids.json"), "r") as f:
        ids = json.load(f)
    return normalize(embeddings, axis=1), ids

# Load pre-generated reformulated queries
def load_reformulated_queries(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 검색 함수
def retrieve(query_text, corpus_embeddings, corpus_ids, raw_corpus, top_k=5):
    query_emb = client.embeddings.create(input=[query_text], model="text-embedding-3-small")
    qvec = np.array(query_emb.data[0].embedding).reshape(1, -1)
    qvec = normalize(qvec, axis=1)
    scores = cosine_similarity(qvec, corpus_embeddings)[0]
    top_indices = scores.argsort()[-top_k:][::-1]
    results = []
    for i in top_indices:
        doc_id = corpus_ids[i]
        doc_data = raw_corpus.get(doc_id, {})
        results.append({
            "Retrieval_Documents_ID": doc_id,
            "Retrieval_Documents_Raw_Text": doc_data.get("text", "")
        })
    return results

# 실행 시작
def main():
    raw_corpus = load_raw_corpus()
    corpus_embeddings, corpus_ids = load_corpus_embeddings()

    file_path="reformulation_queries/reformulation_results.json"
    
    reformulated_queries = load_reformulated_queries(file_path)  # List of dicts
    final_output = []

    for item in reformulated_queries:
        doc_id = item["Doc_ID"]
        original_query = item["Origin_Query"]
        reformulations = item["Reformulation_Queries"]

        for rtype, query_text in zip(["Paraphrasing", "AspectSpecific", "EntityAware"], reformulations):
            if query_text.strip():
                results = retrieve(query_text, corpus_embeddings, corpus_ids, raw_corpus, top_k=TOP_K)
                final_output.append({
                    "Doc_ID": doc_id,
                    "Origin_Query": original_query,
                    "Reformulation_Type": f"{rtype}_Query",
                    "Reformulated_Query": query_text,
                    "Retrieval_Results": results
                })

    # JSON 저장
    with open("retrieval_queries/reformulated_retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print("✅ JSON 저장 완료: retrieval_queries/reformulated_retrieval_results.json")
    return final_output

if __name__ == "__main__":
    main()
