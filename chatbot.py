import os
import re
import json
import torch
import uvicorn
import asyncio
import httpx
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

app = FastAPI()

# PDF 파일 경로 설정
pdf_paths = [
    "정책공약집.pdf",
    "지역공약.pdf"
]

# 전역 변수로 모델과 벡터 스토어 선언
model = None
tokenizer = None
vectorstore = None

def init_rag_system():
    """RAG 시스템 초기화"""
    global model, tokenizer, vectorstore

    print("RAG 시스템 초기화 중...")

    # 1. PDF 문서 로드 및 텍스트 추출
    documents = []
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        else:
            print(f"파일이 존재하지 않습니다: {pdf_path}")

    if not documents:
        print("로드된 문서가 없습니다. 파일 경로를 확인해주세요.")
        return False

    # 2. 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"문서를 {len(chunks)}개의 청크로 분할했습니다.")

    # 3. 임베딩 모델 설정
    print("임베딩 모델 로드 중...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v1",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("임베딩 모델 로드 성공!")

    # 4. 벡터 데이터베이스 생성
    print("벡터 데이터베이스 생성 중...")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    print("벡터 데이터베이스 생성 완료")

    for doc_id in list(vectorstore.docstore._dict.keys())[:3]:  # 처음 3개만 출력해 로그 크기 제한
        print(f"문서 ID {doc_id}의 내용 (샘플):")
        print(vectorstore.docstore._dict[doc_id])
        print("-" * 50)

    # 5. EXAONE 모델 로드
    model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    print(f"{model_name} 모델 로드 중...")

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # 양자화 설정을 BitsAndBytesConfig로 구성
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    # 모델 로드 - 성능 최적화 설정
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("EXAONE 모델 로드 완료!")

    # 성능 최적화 설정 적용
    optimize_performance()

    return True

def optimize_performance():
    """성능 최적화 설정 적용"""
    global model

    # GPU 메모리 최적화
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        torch.cuda.set_device(torch.cuda.current_device())

    # 모델 최적화
    if hasattr(model, 'config'):
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = True
        if hasattr(model.config, 'gradient_checkpointing'):
            model.config.gradient_checkpointing = False

    print("성능 최적화 설정이 적용되었습니다.")

def retrieve_context(query, k):
    """쿼리와 관련된 문서를 검색하여 컨텍스트를 생성합니다."""
    global vectorstore
    relevant_docs = vectorstore.similarity_search(query, k=k)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    return context

def generate_answer(prompt):
    """프롬프트에 대한 응답을 생성합니다."""
    global model, tokenizer

    # 입력 인코딩 - attention_mask 명시적 설정
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    # 효율적인 생성 설정
    generation_config = {
        "max_new_tokens": 500,
        # "do_sample": True,
        # "temperature": temperature,
        "num_beams": 1,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # 토큰 생성
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,  # attention_mask 명시적 전달
            **generation_config
        )

    # 응답 디코딩
    response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def post_process_answer(answer):
    """응답을 후처리하여 일관된 형식으로 정제합니다."""
    # 불필요한 마크다운 제거
    answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)
    
    # "다음과 같습니다" 패턴 제거
    answer = re.sub(r'^(다음과 같습니다|관련 공약은 다음과 같습니다|다음과 같은 내용이 있습니다|다음과 같은 공약이 있습니다|다음을 참고하세요)[\.:]?\s*', '', answer)
    
    # "~입니다"로 시작하는 패턴 제거
    answer = re.sub(r'^[^\.]*입니다[\.:]?\s*', '', answer)
    
    # 모든 콜론 뒤에 줄바꿈 추가
    answer = re.sub(r':\s*', ':\n', answer)
    
    # 번호 리스트 형식 정리
    answer = re.sub(r'(\d+)\.\s+', r'\n\1. ', answer)
    
    # 불필요한 연속된 줄바꿈 정리
    answer = re.sub(r'(<br>){3,}', '<br>', answer)
    
    # 첫 번째 줄이 비어 있으면 제거
    answer = re.sub(r'^(<br>)', '', answer)

    # 항목 앞의 과도한 공백 정리 (- 기호 앞의 공백 없앰)
    answer = re.sub(r'^\s+- ', r'- ', answer, flags=re.MULTILINE)
    
    # 응답이 비어 있는 경우 처리
    if not answer.strip():
        answer = "관련 정보를 찾을 수 없습니다."
        
    return answer.strip()

async def process_llm_and_callback(user_query: str, callback_url: str):
    """
    백그라운드에서 LLM 처리를 수행하고, 완료되면 callbackUrl로 응답을 전송합니다.
    """
    try:
        # 1. RAG 컨텍스트 검색 및 프롬프트 생성 (이 부분은 비교적 빠를 수 있음)
        context = retrieve_context(user_query, k=5)

        prompt = f"""제공한 정보를 바탕으로 사용자 질문에 답하세요.
                  문서 내용에 없는 정보는 추측하지 말고, 정보가 부족하면 솔직히 모른다고 말하세요.

                  ### 응답 형식 ###
                  답변은 다음과 같은 일관된 형식으로 작성하세요:
                  1. 질문 주제와 관련된 공약 또는 정책을 항목별로 나눠 작성합니다.
                  2. 모든 요점 앞에는 번호나 기호(예: 1. 2. 3. 또는 - - -)를 붙여 항목별로 구분합니다.
                  3. 각 항목은 다음 줄에 작성하여 가독성을 높입니다.
                  4. "다음과 같습니다", "~ 입니다", "다음을 참고하세요" 등의 표현으로 시작하지 마세요.
                  5. 각 항목은 짧고 명확한 문장으로 작성합니다.

                  ### 참고 정보: ###
                  {context}

                  ### 사용자 질문 ###
                  {user_query}와 관련된 공약만 문서에 있는 그대로 답변해주세요.

                  ### 답변 ###"""

        # 2. LLM 응답 생성
        print(f"백그라운드 작업 시작: LLM 응답 생성 중... (쿼리: {user_query})")
        raw_answer = await asyncio.to_thread(generate_answer, prompt) # 동기 함수를 스레드에서 비동기 실행
        print(f"백그라운드 작업 완료: LLM 응답 생성 완료. 후처리 시작.")

        # 3. 응답 후처리
        final_answer = post_process_answer(raw_answer)

        print(final_answer)

        print(f"후처리 완료. 콜백 URL로 응답 전송 준비: {callback_url}")

        # 4. callbackUrl로 최종 응답 전송
        # 카카오톡 콜백 응답 형식에 맞춰 JSON 데이터 구성
        callback_res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": f'🌳 {user_query} 관련 답변입니다.\n\n{final_answer}'
                        }
                    }
                ]
            }
            }

        async with httpx.AsyncClient() as client:
            response = await client.post(callback_url, json=callback_res, timeout=60.0)
            print(f"콜백 URL 응답 상태 코드: {response.status_code}")
            response.raise_for_status() # HTTP 오류 발생 시 예외 발생
            print("콜백 URL로 최종 응답 전송 성공.")

    except Exception as e:
        print(f"백그라운드 작업 중 오류 발생: {e}")

        try:
             error_res = {
                "version": "2.0",
                "template": {
                    "outputs": [
                        {
                            "simpleText": {
                                "text": "답변 생성 중 오류가 발생했습니다. 다시 시도해주세요."
                            }
                        }
                    ]
                }
             }
             async with httpx.AsyncClient() as client:
                 await client.post(callback_url, json=error_res, timeout=30.0)
                 print("오류 안내 콜백 전송 성공.")
        except Exception as e_callback:
            print(f"오류 안내 콜백 전송 중 오류 발생: {e_callback}")


# 카카오톡 스킬 엔드포인트 수정
@app.post('/api/chat')
async def kakao_skill(request: Request, background_tasks: BackgroundTasks): # BackgroundTasks 파라미터 추가
    print("카카오톡 스킬 요청 수신")
    req = await request.json()

    # 사용자 발화(query) 추출
    user_query = req['userRequest']['utterance']

    # callbackUrl 추출
    callback_url = req['userRequest'].get('callbackUrl')

    print(f"사용자 쿼리: {user_query}")
    print(f"Callback URL: {callback_url}")

    background_tasks.add_task(process_llm_and_callback, user_query, callback_url)

    print("백그라운드 작업 등록 완료. 즉시 응답 전송.")

    # 카카오톡 서버에 5초 이내로 중간 응답을 반환
    immediate_res = {
        "version" : "2.0",
        "useCallback" : True
    }

    return JSONResponse(content=immediate_res)

@app.on_event("startup")
async def startup_event():
    # 모델 및 RAG 시스템 초기화
    init_success = init_rag_system()

    if not init_success:
        print("초기화 실패. 서버를 시작할 수 없습니다.")
        import sys
        sys.exit(1)
    else:
        print("서버 시작 준비 완료! 카카오톡 스킬 서버를 실행합니다.")

# 서버 실행
if __name__ == "__main__":
    uvicorn.run("chatbot:app", host="0.0.0.0", port=5000)