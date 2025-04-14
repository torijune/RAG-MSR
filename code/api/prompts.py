############## 초기 쿼리 ############
import json
import random
import pandas as pd

def load_queries_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def initial_query():
    queries = load_queries_jsonl("datasets/nfcorpus/nfcorpus_raw/queries.jsonl")
    sampled = random.sample(queries, 3)
    return {q["_id"]: q["text"] for q in sampled}

def get_test_queries(number):
    test_file_path = "datasets/nfcorpus/nfcorpus_raw/qrels/test.tsv"
    query_file_path = "datasets/nfcorpus/nfcorpus_raw/queries.jsonl"

    df_test = pd.read_csv(test_file_path, sep="\t")
    queries = load_queries_jsonl(query_file_path)
    # test_query_id = df_test['query-id']
    # test_answer_id  = df_test['corpus-id']

    test_query_ids = df_test['query-id'].unique().tolist()

    test_query = {q["_id"]: q["text"] for q in queries if q["_id"] in test_query_ids}
    if number == 'all':
        test_query = {q["_id"]: q["text"] for q in queries if q["_id"] in test_query_ids}
    else :
        test_query = dict(list(test_query.items())[:str(number)])
    return test_query

def get_queries_by_ids(query_ids):
    query_file_path = "datasets/nfcorpus/nfcorpus_raw/queries.jsonl"
    queries = load_queries_jsonl(query_file_path)

    selected_queries = {q["_id"]: q["text"] for q in queries if q["_id"] in query_ids}
    return selected_queries


############ GenCRF Reformulation Prompts ############

'''
- Contextual Expansion:
    - 질문의 핵심 의도를 파악한 후, 문맥을 확장하여 더 풍부한 정보를 제공
    - 질문의 문맥을 넓혀 검색 결과를 더 확장하는 방식
- Detail Specific:
    - 질문의 구체적인 세부 사항이나 하위 주제에 초점을 맞춰 답변을 생성
    - 질문의 특정 부분에 대해 깊이 있는 정보를 제공하는 방식
- Aspect Specific:
    - 특정 주제의 한 측면이나 차원에 집중하여 쿼리를 확장하는 방식
    - 주제의 특정 측면에 대해 더 구체적이고 풍부한 결과를 도출할 수 있도록 도움
'''

def ContextualExpansion(query):
    prompt = f"""
    You are a contextual expansion expert. Your task is to understand the core intent of the original query and provide a refined, contextually expanded answer within 100 characters.
    Below is the query: {query}
    """
    return prompt

def DetailSpecific(query):
    prompt = f"""
    You are a detail-specific expert. Your task is to understand the core intent of the original query and provide a refined, detailed answer focusing on particular details or subtopics within 100 characters.
    Below is the query: {query}
    """
    return prompt

def GenCRF_AspectSpecific(query):
    prompt = f"""
    You are an aspect-specific inquiry expert. Your task is to provide a refined answer focusing on a specific aspect or dimension within 100 characters.
    Below is the query: {query}
    """
    return prompt

############ RAG-MSR Reformulation Prompt ############

'''
- Paraphrasing:
    - 쿼리를 다른 표현으로 바꾸되 의미는 동일하게 유지
    - 다양한 표현을 통해 검색 다양성을 높이는 데 유리
- AspectSpecific:
    - 특정 측면(예: 비용, 성능, 사용성 등)에 초점을 맞춰 쿼리를 재작성
    - 검색 범위를 좁혀 더 정밀한 정보를 얻고자 할 때 유용
- EntityAware:
    - 쿼리에 사람, 장소, 도구, 이벤트 등 명시적 엔티티를 포함시켜 재작성
    - 검색 시스템의 정확도를 높이는 데 효과적
'''

# 문장이 너무 짧을때는 해결하기 위해 추가
input_query_type_rule = """
Analyze the given query and determine its characteristics.

Depending on the query type, decide the most effective reformulation strategy for improving retrieval accuracy.

Consider:
- If the query is too short or keyword-based 
    → Expand into a natural language question with background context and the reformulation rule.
- If the query is already a full sentence 
    → Follow the reformulation rule below to refine a full sentence.
"""

output_rule = """
Output Rules:
- Do not explain your strategy.
- Only output the final reformulated query within 100 characters .
"""

# Group_1
def Paraphrasing(query):
    prompt = f"""
    You are a paraphrasing expert. 

    Your task is to rewrite the original query in a new way using different wording but maintaining the same meaning. 
    Original Query: {query}
    """
    return prompt

def RAG_MSR_AspectSpecific(query):
    prompt = f"""
    You are an aspect-specific reformulation expert. 

    - Reformulation Rule:
        Your task is to rewrite the query focusing on one specific dimension (e.g., cost, usability, performance, user experience). 
    Original Query: {query}
    """
    return prompt

def EntityAware(query):
    prompt = f"""
    You are an entity-aware reformulation expert. 
    - Reformulation Rule:
        Rewrite the query by incorporating specific entities (e.g., people, places, tools, events) to improve retrieval accuracy. 

    Original Query: {query}
    """
    return prompt


# Group_2
def Clarification_Reformulation(query):
    prompt = f"""You are an expert search assistant.
    {input_query_type_rule}

    - Reformulation Rule: 
        Rewrite the query to clarify intent, resolve ambiguity, and add necessary context.
    {output_rule}

    Original Query: "{query}"
    """
    return prompt


def EntityExpansion_Reformulation(query):
    prompt = f"""You are an expert in query expansion.
    {input_query_type_rule}

    - Reformulation Rule:   
        Rewrite the query by adding important entities, keywords, or related concepts to improve retrieval performance.
    {output_rule}

    Original Query: "{query}"
    """
    return prompt


def RetrievalCondense_Reformulation(query):
    prompt = f"""You are an expert in information retrieval optimization.
    {input_query_type_rule}

    - Reformulation Rule: 
        Rewrite the query in a concise, document-friendly style suitable for retrieval, preserving only essential keywords.
    {output_rule}

    Original Query: "{query}"
    """
    return prompt

########## RAG-MSR LLM Scoring Prompts ########

'''
LLM Scoring 프롬프트
LLM Scoring은 각 클러스터의 쿼리들을 평가하여 점수를 부여하는 프롬프트입니다.
관련성, 구체성, 명확성, 포괄성, 유용성을 기준으로 점수를 매깁니다.
'''

def LLMScoring(q_init, rerank_queries):
    # q_init는 그냥 쿼리, rerank_queries는 리랭킹된 문서 집합(딕셔너리)
    formatted_docs = "\n".join([f"Doc{i+1}: {doc}" for i, doc in enumerate(rerank_queries)])
    prompt = f"""
    You are an expert in information retrieval evaluation. Your task is to score each reranked document based on how well it addresses the user's original question. Consider the following aspects:

    - Relevance to the user's intent
    - Clarity and usefulness of the content
    - Completeness and specificity of the answer
    - Overall alignment with the original query

    Assign a score from 0 to 100 for each document. A document that fully addresses the question in a clear and specific way should receive a high score (90–100), while a document that is vague, irrelevant, or only partially related should score lower.

    Only return the scores in the following list format:
    [score_doc1, score_doc2, score_doc3, score_doc4, score_doc5]

    Original Query: {q_init}
    Reranked Documents:
    {formatted_docs}
    """
    return prompt


########## RAG-MSR Reformulation으로 해결할 수 있는 기존 한계점 ##########

'''
🔍 기존 RAG 기반 연구의 한계점
	1.	재질문(Query)의 다양성이 부족해 검색된 문서가 편향될 수 있음
	2.	사용자 쿼리가 모호하거나 광범위할 경우 관련 문서 검색 실패 가능성
	3.	정확한 정보 검색보다 단순 연관 키워드 매칭에 의존
	4.	명확한 의도 파악 없이 문맥을 확장해 버려 irrelevant한 문서가 선택됨
	5.	단일 질의로 인해 다양한 관점의 정보 탐색 부족

✅ RAG-MSR 프롬프트들이 해결하는 방식
- Paraphrasing:
    - 1. 질의 다양성 부족 해결
- AspectSpecific: 
    - 2,3,5. 포괄적 쿼리에서 특정 측면에 집중하여 초점 있는 쿼리로 재구성
- EntityAware: 
    - 2,4,5. 문맥 부족 문제를 명시적 entity를 넣음으로써 개선
'''