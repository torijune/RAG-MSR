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
from api.utils import save_json, load_data, load_answer

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def RAG_MSR(run: int, use_reformulation=True, use_reranking=True):
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
            result = reformulation(doc_id, query)
            logger.info(f"재구성된 쿼리 목록 for {doc_id}:")
            for q in result["Reformulation_Queries"]:
                logger.info(" - %s", q)
            reformulation_results.append(result)
        reformulation_output_dir = "reformulation_queries"
        reformulation_file_name = f"reformulation_results_{run}.json"
        save_json(reformulation_results, reformulation_output_dir, reformulation_file_name)
        logger.info("✅ Query Reformulation complete!")
    else:
        reformulation_results = [{"Doc_ID": doc_id, "Origin_Query": query, "Reformulation_Queries": [query, query, query]} for doc_id, query in query_dict.items()]
        reformulation_output_dir = "reformulation_queries"
        reformulation_file_name = f"reformulation_results_{run}.json"
        save_json(reformulation_results, reformulation_output_dir, reformulation_file_name)

    # Retrieval Step

    logger.info("🕘 Start Documents Retrieval ...")

    raw_corpus = load_raw_corpus()
    corpus_embeddings, corpus_ids = load_corpus_embeddings()

    reformulated_queries = load_reformulated_queries(os.path.join(reformulation_output_dir, reformulation_file_name))
    retrieval_results = []

    for item in reformulated_queries:
        doc_id = item["Doc_ID"]
        original_query = item["Origin_Query"]
        reformulations = item["Reformulation_Queries"]

        for rtype, query_text in zip(["Paraphrasing", "AspectSpecific", "EntityAware"], reformulations):
            if query_text.strip():
                results = retrieve(query_text, corpus_embeddings, corpus_ids, raw_corpus, top_k=5)
                retrieval_results.append({
                    "Doc_ID": doc_id,
                    "Origin_Query": original_query,
                    "Reformulation_Type": f"{rtype}_Query",
                    "Reformulated_Query": query_text,
                    "Retrieval_Results": results
                })
    retrieval_output_dir = "retrieval_queries"
    retrieval_file_name = f"retrieval_results_{run}.json"

    save_json(retrieval_results, retrieval_output_dir, retrieval_file_name)

    logger.info("✅ Documents Retrieval complete!")

    # Reranking Step
    if use_reranking:
        logger.info("🕘 Start Documents Reranking ...")
        input_data = load_data(os.path.join(retrieval_output_dir, retrieval_file_name))
        data = group_by_doc_id(input_data)
        rerank_results = rerank_documents(data)
        reranking_output_dir = "reranked_queries"
        reranking_file_name = f"reranked_results_{run}.json"
        save_json(rerank_results, reranking_output_dir, reranking_file_name)
        logger.info("✅ Documents Reranking complete!")
        predictions = load_predictions(os.path.join(reranking_output_dir, reranking_file_name))
    else:
        predictions = load_predictions(os.path.join(retrieval_output_dir, retrieval_file_name))

    # Evaluation Step
    logger.info("🕘 Start Documents Evaluation ...")

    output_path = f"prediction/predictions_{run}.jsonl"
    save_predictions(predictions, output_path)

    qrels_path = "datasets/nfcorpus/nfcorpus_raw/qrels/test.tsv"
    df_answer, answer_query_id, answer_corpus_id = load_answer(qrels_path)
    eval_results = evaluate_all_with_predictions(predictions, df_answer, answer_query_id, answer_corpus_id)

    evaluation_output_dir = "Nfcorpus_nDCG"
    evaluation_file_name = f"evaluation_results_{run}.json"

    save_json(eval_results, evaluation_output_dir, evaluation_file_name)
    logger.info("✅ Documents Evaluation complete!")

def multi_round_RAG_MSR():
    num_runs = 1
    ablation_settings = [
        (False, False),  # Raw Retrieval
        (True, False),   # Reformulation Retrieval
        (False, True),   # Reranking Retrieval
        (True, True),    # Origin RAG-MSR
    ]
    for run, (use_reformulation, use_reranking) in enumerate(ablation_settings, start=1):
        print(f"\n===== RAG-MSR Execution Round {run} - Reformulation: {use_reformulation}, Reranking: {use_reranking} =====")
        RAG_MSR(run, use_reformulation, use_reranking)
        print(f"===== Finished RAG-MSR Execution Round {run} =====")

if __name__ == "__main__":
    # multi_round_RAG_MSR
    multi_round_RAG_MSR()