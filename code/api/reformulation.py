import json
import os
from typing import List
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from api.prompts import initial_query

'''
GenCRF : ContextualExpansion, DetailSpecific, GenCRF_AspectSpecific
RAG-MSR : Paraphrasing, RAG_MSR_AspectSpecific, EntityAware
'''

# Ollama 모델 설정
llm = ChatOllama(model="llama-3.1-8B-instrcut:latest", top_p=0.95, num_predict=100)


# GenCRF의 Multi-Reformulation 진행 함수
def GenCRF_generate_reformulated_queries(user_query):
    """
    사용자 쿼리를 기반으로 각 프롬프트에서 2개의 쿼리를 생성하는 함수

    :param user_query: 사용자가 입력한 초기 쿼리
    :return: 각 프롬프트에서 생성된 쿼리들 (딕셔너리 형태)
    """
    from api.prompts import ContextualExpansion, DetailSpecific, GenCRF_AspectSpecific
    # 각각의 프롬프트를 ChatPromptTemplate으로 생성
    CE_prompt = ChatPromptTemplate.from_template(ContextualExpansion(user_query))
    DS_prompt = ChatPromptTemplate.from_template(DetailSpecific(user_query))
    AS_prompt = ChatPromptTemplate.from_template(GenCRF_AspectSpecific(user_query))

    # 재구성된 쿼리를 저장할 딕셔너리
    reform_queries = {}

    # 프롬프트와 그에 맞는 딕셔너리 키를 설정
    prompts = [("CE_prompt", CE_prompt), ("DS_prompt", DS_prompt), ("AS_prompt", AS_prompt)]

    # 각 프롬프트를 처리하여 2개의 쿼리 생성 및 결과 저장
    for name, prompt in prompts:
        reform_queries[name] = []
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"prompts": prompt})  # ⚠️ prompt → 실제 변수에 맞게 수정 필요
        reform_queries[name].append(result)

    return reform_queries

# RAG-MSR의 Multi-Reformulation 진행 함수
def RAG_MSR_generate_reformulated_queries(user_query, reformulatio_group):
    """
    RAG-MSR용 프롬프트 세트를 사용하여 재구성 쿼리를 생성하는 함수

    :param user_query: 사용자의 초기 쿼리
    :return: Paraphrasing, AspectSpecific, EntityAware 프롬프트를 통해 생성된 재구성 쿼리 딕셔너리
    """
    from api.prompts import Paraphrasing, RAG_MSR_AspectSpecific, EntityAware
    # group_1
    para_prompt = ChatPromptTemplate.from_template(Paraphrasing(user_query))
    asp_prompt = ChatPromptTemplate.from_template(RAG_MSR_AspectSpecific(user_query))
    ent_prompt = ChatPromptTemplate.from_template(EntityAware(user_query))
    
    from api.prompts import Clarification_Reformulation, EntityExpansion_Reformulation, RetrievalCondense_Reformulation
    # group_2
    CR_prompt = ChatPromptTemplate.from_template(Clarification_Reformulation(user_query))
    EER_prompt = ChatPromptTemplate.from_template(EntityExpansion_Reformulation(user_query))
    RC_prompt = ChatPromptTemplate.from_template(RetrievalCondense_Reformulation(user_query))

    # 재구성된 쿼리를 저장할 딕셔너리
    reform_queries = {}

    # 프롬프트와 그에 맞는 딕셔너리 키를 설정
    # group_1
    prompts_group_1 = [("Paraphrasing", para_prompt), ("AspectSpecific", asp_prompt), ("EntityAware", ent_prompt)]
    # group_2
    prompts_group_2 = [("Clarification_Reformulation", CR_prompt), ("EntityExpansion_Reformulation", EER_prompt), ("RetrievalCondense_Reformulation", RC_prompt)]

    # 선택된 그룹에 따라 프롬프트 선택
    if reformulatio_group == "group_1":
        final_prompts = prompts_group_1
    elif reformulatio_group == "group_2":
        final_prompts = prompts_group_2
    else:
        raise ValueError("Invalid reformulatio_group. Choose 'group_1' or 'group_2'.")

    # 각 프롬프트를 처리하여 1개의 쿼리 생성 및 결과 저장
    for name, prompt in final_prompts:
        reform_queries[name] = []
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({})
        reform_queries[name].append(result)

    return reform_queries

def save_reformulated_queries(result: List[dict], output_dir: str, file_name: str):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def main(doc_id: str, query: str, reformulatio_group: str) -> dict:
    """
    사용자 입력을 기반으로 LLM에서 생성한 다중 질의 딕셔너리 반환
    :param doc_id: 쿼리의 원본 ID
    :param query: 쿼리 텍스트
    :param reformulatio_group: 선택된 리포뮬레이션 그룹
    :return: {Doc_ID: doc_id, Origin_Query: query, Reformulation_Queries: [reform_1, reform_2, ...]}
    """
    reform_dict = RAG_MSR_generate_reformulated_queries(query, reformulatio_group)
    queries = []
    for qlist in reform_dict.values():
        queries.extend(qlist)

    return {
        "Doc_ID": doc_id,
        "Origin_Query": query,
        "Reformulation_Queries": queries
    }

# 테스트용
if __name__ == "__main__":
    from api.prompts import get_test_queries  # get_test_queries를 사용할거니까

    query_dict = get_test_queries()  # Dict 형태: {_id: text}
    all_results = []

    reformulatio_group = "group_1"  # 선택할 group 설정

    for doc_id, query in query_dict.items():
        result = main(doc_id, query, reformulatio_group)

        print(f"재구성된 쿼리 목록 for {doc_id}:")
        for q in result["Reformulation_Queries"]:
            print(" -", q)

        all_results.append(result)
    
    output_dir: str = f"reformulation_queries/{reformulatio_group}"
    file_name: str = "reformulation_results.json"

    save_reformulated_queries(all_results, output_dir, file_name)