import json
import os
import pandas as pd
from typing import List, Dict

def save_json(data: List[dict], output_dir: str, file_name: str):
    """Saves the given data as a JSON file in the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_data(input_json_path: str) -> List[Dict]:
    """Loads data from a JSON file and returns it as a list of dictionaries."""
    with open(input_json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_answer(file_path: str):
    """Loads answers from a TSV file and returns the DataFrame along with query and corpus IDs."""
    df_answer = pd.read_csv(file_path, sep="\t")
    answer_query_id = df_answer["query-id"]
    answer_corpus_id = df_answer["corpus-id"]
    return df_answer, answer_query_id, answer_corpus_id

def load_predictions_reranking(data):
    predictions_dict = {}

    # 리스트 형태일 경우 처리
    if isinstance(data, list):
        for item in data:
            query_id = item["Doc_ID"]
            reranked_results = item["Reranked_Results"]
            reranked_ids = [doc["Retrieval_Documents_ID"] for doc in reranked_results]

            if query_id in predictions_dict:
                predictions_dict[query_id].extend(reranked_ids)
            else:
                predictions_dict[query_id] = reranked_ids
    else:
        for doc_id, items in data.items():
            for item in items:
                query_id = item["Doc_ID"]
                reranked_results = item["Reranked_Results"]
                reranked_ids = [doc["Retrieval_Documents_ID"] for doc in reranked_results]

                if query_id in predictions_dict:
                    predictions_dict[query_id].extend(reranked_ids)
                else:
                    predictions_dict[query_id] = reranked_ids

    predictions = []
    for qid, doc_ids in predictions_dict.items():
        unique_doc_ids = list(dict.fromkeys(doc_ids))
        predictions.append({"qid": qid, "doc_ids": unique_doc_ids})

    return predictions

def load_predictions_retrieval(data):
    predictions_dict = {}

    # 리스트 형태일 경우 처리
    if isinstance(data, list):
        for item in data:
            query_id = item["Doc_ID"]
            retrieval_results = item["Retrieval_Results"]
            retrieval_ids = [doc["Retrieval_Documents_ID"] for doc in retrieval_results]

            if query_id in predictions_dict:
                predictions_dict[query_id].extend(retrieval_ids)
            else:
                predictions_dict[query_id] = retrieval_ids
    else:
        for doc_id, items in data.items():
            for item in items:
                query_id = item["Doc_ID"]
                retrieval_results = item["Retrieval_Results"]
                retrieval_ids = [doc["Retrieval_Documents_ID"] for doc in retrieval_results]

                if query_id in predictions_dict:
                    predictions_dict[query_id].extend(retrieval_ids)
                else:
                    predictions_dict[query_id] = retrieval_ids

    predictions = []
    for qid, doc_ids in predictions_dict.items():
        unique_doc_ids = list(dict.fromkeys(doc_ids))
        predictions.append({"qid": qid, "doc_ids": unique_doc_ids})

    return predictions