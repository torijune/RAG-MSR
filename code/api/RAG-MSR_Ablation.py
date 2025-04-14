import json
import os
import pandas as pd
from typing import List, Dict
import logging

from api.prompts import get_queries_by_ids
from api.reformulation import main as reformulation
from api.retrieval import load_raw_corpus, load_corpus_embeddings, load_reformulated_queries, retrieve
from api.reranking import group_by_doc_id, rerank_documents
from api.evaluation.evaluation import load_predictions, save_predictions, evaluate_all_with_predictions
from api.utils import save_json, load_data, load_answer, load_predictions_reranking, load_predictions_retrieval

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def RAG_MSR(use_reformulation, use_reranking, reformulation_group: str):
    """Main function to execute the RAG-MSR process including query reformulation, retrieval, reranking, and evaluation."""
    # input 쿼리 정의
    query_ids = ["PLAIN-2", "PLAIN-33", "PLAIN-56", "PLAIN-68", "PLAIN-91", "PLAIN-102",
                  "PLAIN-12", "PLAIN-44", "PLAIN-23", "PLAIN-78"]
    query_dict = get_queries_by_ids(query_ids)  # Dict 형태: {_id: text}
    logger.info("✅ BeIR nfcorpus load complete!")

    # Reformulation Step
    if use_reformulation:
        reformulation_results = []
        logger.info("🕘 Start Query Reformulation ...")
        for doc_id, query in query_dict.items():
            result = reformulation(doc_id, query, reformulation_group)
            logger.info(f"재구성된 쿼리 목록 for {doc_id}:")
            for q in result["Reformulation_Queries"]:
                logger.info(" - %s", q)
            reformulation_results.append(result)
        logger.info("✅ Query Reformulation complete!")
    else:
        reformulation_results = [{"Doc_ID": doc_id, "Origin_Query": query, "Reformulation_Queries": [query, query, query]} for doc_id, query in query_dict.items()]

    # Retrieval Step

    logger.info("🕘 Start Documents Retrieval ...")

    raw_corpus = load_raw_corpus()
    corpus_embeddings, corpus_ids = load_corpus_embeddings()

    reformulated_queries = reformulation_results
    retrieval_results = []

    for item in reformulated_queries:
        doc_id = item["Doc_ID"]
        original_query = item["Origin_Query"]
        reformulations = item["Reformulation_Queries"]
        
        # Reformulation Type Group에 맞춰 수정
        if reformulation_group == "group_1":
            reformulation_group = ["Paraphrasing", "AspectSpecific", "EntityAware"]
        elif reformulation_group == "group_2":
            reformulation_group = ["Clarification", "EntityExpansion", "RetrievalCondense"]

        for rtype, query_text in zip(reformulation_group, reformulations):
            if query_text.strip():
                results = retrieve(query_text, corpus_embeddings, corpus_ids, raw_corpus, top_k=5)
                retrieval_results.append({
                    "Doc_ID": doc_id,
                    "Origin_Query": original_query,
                    "Reformulation_Type": f"{rtype}_Query",
                    "Reformulated_Query": query_text,
                    "Retrieval_Results": results
                })
    logger.info("✅ Documents Retrieval complete!")

    # Reranking Step
    if use_reranking:
        logger.info("🕘 Start Documents Reranking ...")
        data = group_by_doc_id(retrieval_results)
        rerank_results = rerank_documents(data)
        logger.info("✅ Documents Reranking complete!")
        predictions = load_predictions_reranking(rerank_results)
    else:
        predictions = load_predictions_retrieval(retrieval_results)

    # Evaluation Step
    logger.info("🕘 Start Documents Evaluation ...")

    qrels_path = "datasets/nfcorpus/nfcorpus_raw/qrels/test.tsv"
    df_answer, answer_query_id, answer_corpus_id = load_answer(qrels_path)
    eval_results = evaluate_all_with_predictions(predictions, df_answer, answer_query_id, answer_corpus_id)

    logger.info("✅ Documents Evaluation complete!")

    return {
        "reformulation_results": reformulation_results,
        "retrieval_results": retrieval_results,
        "rerank_results": predictions,
        "eval_results": eval_results
    }

def RAG_MSR_Ablation_Experiment():
    ablation_settings = [
        # 그냥 retrieval 만
        (False, False),
        # reformulation하고 retrieval
        (True, False),
        # retrieval하고 rerank
        (False, True),
        # 둘다
        (True, True),
    ]

    ablation_names = [
        "raw_retrieval",
        "reformulation_retrieval",
        "reranking_retrieval",
        "origin_rag_msr"
    ]

    for (use_reformulation, use_reranking), run_name in zip(ablation_settings, ablation_names):
        reformulation_group="group_2"
        print(f"\n===== RAG-MSR Execution - {run_name.replace('_', ' ').title()} =====")
        results = RAG_MSR(use_reformulation=use_reformulation, use_reranking=use_reranking, reformulation_group=reformulation_group)

        save_json(results["reformulation_results"], f"Ablation/{run_name}_{reformulation_group}", "reformulation_results.json")
        save_json(results["retrieval_results"], f"Ablation/{run_name}_{reformulation_group}", "retrieval_results.json")
        save_json(results["rerank_results"], f"Ablation/{run_name}_{reformulation_group}", "reranked_results.json")
        save_predictions(results["rerank_results"] or results["retrieval_results"], f"Ablation/{run_name}_{reformulation_group}", "predictions.jsonl")
        save_json(results["eval_results"], f"Ablation/{run_name}_{reformulation_group}", "evaluation_results.json")
        print(f"===== Finished RAG-MSR Execution - {run_name.replace('_', ' ').title()} =====")

if __name__ == "__main__":
    RAG_MSR_Ablation_Experiment()