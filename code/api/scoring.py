from typing import List, Dict, Tuple
import random
import os
import json
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from api.prompts import LLMScoring

load_dotenv()  # .env 파일에서 환경변수 로드

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
def CosineScoring(initial_query, reranked_queries) -> List[float]:
    try:
        inputs = [initial_query] + reranked_queries
        response = client.embeddings.create(input=inputs, model="text-embedding-3-small")
        embeddings = [np.array(e.embedding) for e in response.data]
        norm_embeddings = normalize(embeddings)
        # initial_query가 첫번째 리스트니까 0번째
        query_embedding = norm_embeddings[0]
        # 나머지 임베딩이 그 뒤의 reranking 쿼리 임베딩
        doc_embeddings = norm_embeddings[1:]

        scores = cosine_similarity([query_embedding], doc_embeddings)[0]
        norm_scores = [round(score, 3) for score in scores]
        return norm_scores
    except Exception as e:
        print(f"CosineScoring Error: {e}")
        return [0.0 for _ in reranked_queries]
import ast  # 리스트 파싱용

def LLMScoring(initial_query, reranked_queries) -> List[float]:
    try:
        from api.prompts import LLMScoring as build_prompt  # 이름 구분
        prompt = build_prompt(initial_query, reranked_queries)

        system_msg = (
            "You are an expert in information retrieval evaluation."
            "Assign a score from 0 to 100 for each document. A document that fully addresses the question in a clear and specific way should receive a high score (90–100), while a document that is vague, irrelevant, or only partially related should score lower."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            top_p=0.95
        )

        score_text = response.choices[0].message.content.strip()
        scores = ast.literal_eval(score_text)  # parse list safely
        scores = [round(float(s) / 100.0, 3) for s in scores]
        return scores
    except Exception as e:
        print(f"LLMScoring Error: {e}")
        return [0.0 for _ in reranked_queries]


def load_reranked_results(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)

def save_filtered_results(data, output_path="reranked_queries/scored_filtered_results.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def score_documents(data, alpha=0.4, threshold=0.5):
    filtered_data = {}

    for doc_id in data.keys():
        for item in data[doc_id]:
            current_doc_id = item["Doc_ID"]
            if current_doc_id not in filtered_data:
                filtered_data[current_doc_id] = []

            origin_query = item["Origin_Query"]
            reranked_results = item["Reranked_Results"]

            doc_ids = [doc["Retrieval_Documents_ID"] for doc in reranked_results]
            doc_texts = [doc["Retrieval_Documents_Raw_Text"] for doc in reranked_results]

            cosine_scores = CosineScoring(origin_query, doc_texts)
            llm_scores = LLMScoring(origin_query, doc_texts)

            final_scores = [round(alpha * c + (1 - alpha) * l, 3) for c, l in zip(cosine_scores, llm_scores)]

            selected = [(doc_id, doc_text, c, l, f) for doc_id, doc_text, c, l, f in
                        zip(doc_ids, doc_texts, cosine_scores, llm_scores, final_scores) if f >= threshold]

            print(f"\nDoc_ID: {item['Doc_ID']} / Reformulation Type: {item['Reformulation_Type']}")
            new_results = []
            for doc_id, doc_text, c_score, l_score, f_score in selected:
                print(f"- {doc_id} | Cosine: {c_score}, LLM: {l_score}, Final: {f_score}")
                new_results.append({
                    "Retrieval_Documents_ID": doc_id,
                    "Retrieval_Documents_Raw_Text": doc_text,
                    "Cosine_Score": c_score,
                    "LLM_Score": l_score,
                    "Final_Score": f_score
                })

            item["Filtered_Results"] = new_results
            filtered_data[current_doc_id].append(item)

    return filtered_data

def main():
    file_path = "reranked_queries/reranked_all_results.json"
    alpha = 0.4
    threshold = 0.2

    data = load_reranked_results(file_path)
    filtered_data = score_documents(data, alpha, threshold)
    save_filtered_results(filtered_data)

if __name__ == "__main__":
    main()