import json
from typing import List, Dict
from collections import defaultdict
from sentence_transformers import CrossEncoder


def load_input_data(input_json_path: str) -> List[Dict]:
    with open(input_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_by_doc_id(input_data: List[Dict]) -> Dict[str, List[Dict]]:
    data = defaultdict(list)
    for item in input_data:
        doc_id = item["Doc_ID"]
        data[doc_id].append(item)
    return dict(data)


def rerank_documents(data: Dict[str, List[Dict]], model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 2) -> Dict[str, List[Dict]]:
    model = CrossEncoder(model_name)
    for doc_id, queries in data.items():
        for query in queries:
            origin_query = query["Origin_Query"]
            retrieval_docs = query["Retrieval_Results"]

            doc_ids = [doc["Retrieval_Documents_ID"] for doc in retrieval_docs]
            doc_texts = [doc["Retrieval_Documents_Raw_Text"] for doc in retrieval_docs]
            pairs = [[origin_query, doc_text] for doc_text in doc_texts]

            scores = model.predict(pairs)

            min_score, max_score = min(scores), max(scores)
            if max_score > min_score:
                scores = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                scores = [0.5 for _ in scores]

            reranked = sorted(zip(doc_ids, doc_texts, scores), key=lambda x: x[2], reverse=True)[:top_k]

            rerank_results = [{"Retrieval_Documents_ID": doc_id, "Retrieval_Documents_Raw_Text": doc_text} for doc_id, doc_text, score in reranked]
            query["Reranked_Results"] = rerank_results
    return data


def save_results(data: Dict[str, List[Dict]], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    input_json_path = "retrieval_queries/reformulated_retrieval_results.json"
    output_json_path = "reranked_queries/reranked_all_results.json"

    input_data = load_input_data(input_json_path)
    data = group_by_doc_id(input_data)
    data = rerank_documents(data)
    save_results(data, output_json_path)

    print("모든 Doc_ID에 대해 Reranking 완료 및 저장 완료!")