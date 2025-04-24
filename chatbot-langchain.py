import os
import re
import json
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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

# 전역 변수로 LangChain 구성요소 선언
retriever = None
llm = None
rag_chain = None

# 1. 검색 최적화를 위한 쿼리 재구성 프롬프트
query_transformation_prompt = PromptTemplate.from_template(
    """주어진 질문을 분석하고, 관련 문서를 효과적으로 검색하기 위한 최적의 검색어만 출력. 설명은 불필요.
    
    원래 질문: {question}

    검색어 : 
    """
)

# 2. 응답 생성 프롬프트
answer_generation_prompt = PromptTemplate.from_template(
    """1. 문서에서 질문과 관련된 정책/공약 제목 추출
    2. 관련 정책이 없다면 "관련 정책 정보 없음"이라고 답
    3. 추출된 정보를 바탕 2~3가지 공약을 JSON 형식으로 출력
    
    질문: {question}
    
    문서 내용:
    {context}
    
    JSON 형식은 다음과 같아야 함 : 
    ```json
    {{
        "공약": [ "첫번째 공약 제목 또는 핵심 키워드", "두번째 공약 제목", ... ],
    }}
    ```

    JSON 답변 : 
    """
)

# 전체 멀티모달 RAG 체인 구성
def create_multimodal_rag_chain(retriever, llm):
    # 1. 쿼리 변환 체인
    query_transformer_chain = (
        {"question": RunnablePassthrough()}
        | query_transformation_prompt
        | llm
        | StrOutputParser()
    )
    
    # 검색 체인
    def retrieve_documents(input_dict):
        original_question = input_dict["original_question"]
        optimized_query = input_dict["optimized_query"]
        
        # 최적화된 쿼리로 검색
        retrieved_docs = retriever.invoke(optimized_query)
        
        return {
            "question": original_question,
            "context": "\n\n".join([doc.page_content for doc in retrieved_docs])
        }
    
    # 응답 생성 체인
    answer_chain = (
        answer_generation_prompt
        | llm
        | StrOutputParser()
    )
    
    # 후처리 함수 정의 - JSON 응답 처리
    def format_answer(answer):
        print(answer)
        json_pattern = r'JSON 답변 :\s*```(?:json)?\s*([\s\S]*?)```'
        json_match = re.search(json_pattern, answer)
        
        if json_match:
            # JSON 코드 블록 내용 추출
            json_str = json_match.group(1).strip()
            return json_str
        
        return None
    
    # 전체 멀티모달 체인 구성
    multimodal_chain = (
        # 1단계: 원본 질문 저장 및 쿼리 최적화
        RunnablePassthrough().with_config(run_name="Original Question")
        | {"original_question": lambda x: x, "optimized_query": query_transformer_chain}
        
        # 2단계: 최적화된 쿼리로 문서 검색
        | RunnableLambda(retrieve_documents).with_config(run_name="Document Retrieval")
        
        # 3단계: 통합된 문서 분석 및 최종 응답 생성 (2번과 3번 단계 통합)
        | answer_chain
        
        # 4단계: 응답 후처리
        | RunnableLambda(format_answer)
    )
    
    return multimodal_chain

def init_rag_system():
    """LangChain RAG 시스템 초기화"""
    global retriever, llm, rag_chain

    print("LangChain RAG 시스템 초기화 중...")

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

    # 샘플 문서 확인 (디버깅용)
    for i, doc_id in enumerate(list(vectorstore.docstore._dict.keys())[:3]):  # 처음 3개만 출력
        print(f"문서 ID {doc_id}의 내용:")
        print(vectorstore.docstore._dict[doc_id])
        print("-" * 50)
        if i >= 2:  # 최대 3개만 출력
            break

    # 5. 검색기(Retriever) 설정 - 벡터스토어에서 직접 생성
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    # 6. EXAONE 모델 로드 및 LangChain LLM 래퍼 설정
    model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
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

    # 모델 최적화 설정 적용
    optimize_performance(model)

    # 직접 Hugging Face 파이프라인 생성 후 LangChain 래퍼 적용
    from transformers import pipeline

    # 트랜스포머 파이프라인 생성
    hf_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        # do_sample=True,
        # temperature=0.3,
        device_map="auto",
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    # LangChain HuggingFacePipeline 래퍼 생성
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    print("EXAONE 모델 로드 완료!")

    rag_chain = create_multimodal_rag_chain(retriever, llm)

    print("LangChain RAG 시스템 초기화 완료!")
    return True

def optimize_performance(model):
    """성능 최적화 설정 적용"""
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

# 기본 경로 테스트용
@app.get('/')
def hello_world():
    return '카카오톡 챗봇 LLM 서버가 실행 중입니다!'

# 카카오톡 스킬 엔드포인트
@app.post('/api/chat')
async def kakao_skill(request: Request):
    global rag_chain
    
    # 카카오톡 스킬 요청 파싱
    req = await request.json()

    # 사용자 발화(query) 추출
    user_query = req['userRequest']['utterance']

    # 멀티모달 LangChain 체인으로 응답 생성
    answer = rag_chain.invoke(user_query)

    # JSON 문자열 파싱
    parsed_json = json.loads(answer)

    # 공약 목록 추출
    policies = parsed_json.get("공약", [])

    # 카카오톡 스킬 응답 형식으로 변환
    if policies:
        # 공약이 있는 경우
        description = ""
        for idx, policy in enumerate(policies, 1):
            description += f"{idx}. {policy}\n"
    else:
        # 공약이 없는 경우
        description = parsed_json.get("메시지", "요청하신 내용에 대한 정보를 찾을 수 없습니다")

    res = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "textCard": {
                        "title": f'🤖 {user_query} 관련 답변드립니다.',
                        "description": description,
                        "buttons": [
                            {
                                "action": "webLink",
                                "label": "🌍 지역공약 살펴보기",
                                "webLinkUrl": "https://www.naver.com/"
                            },
                            {
                                "action": "webLink",
                                "label": "📚 정책공약집 살펴보기",
                                "webLinkUrl": "https://www.naver.com/"
                            }
                        ]
                    }
                }
            ]
        }
    }

    print(res)

    return res

# 서버 초기화 및 실행을 위한 이벤트
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
    uvicorn.run("chatbot-langchain:app", host="0.0.0.0", port=5000)
                                                                                      
