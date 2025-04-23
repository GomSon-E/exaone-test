import os
import re
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = FastAPI()

# PDF 파일 경로 설정 (기존 코드와 동일)
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

    for doc_id in list(vectorstore.docstore._dict.keys()):
        print(f"문서 ID {doc_id}의 내용:")
        print(vectorstore.docstore._dict[doc_id])
        print("-" * 50)

    # 5. EXAONE 모델 로드
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

def generate_answer(prompt, max_new_tokens, temperature):
    """프롬프트에 대한 응답을 생성합니다."""
    global model, tokenizer

    # 입력 인코딩 - attention_mask 명시적 설정
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    # 효율적인 생성 설정 - 수정된 부분
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
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

def answer_with_rag(query, k=5, max_tokens=100, temperature=0.5):
    """RAG로 컨텍스트를 검색하고 응답을 생성합니다."""
    context = retrieve_context(query, k=k)

    # 간결한 프롬프트 구성
    prompt = f""" 제공한 정보를 바탕으로 사용자 질문에 답하세요.
                문서 내용에 없는 정보는 추측하지 말고, 정보가 부족하면 솔직히 모른다고 말하세요.
                사용자 요청 키워드를 답변에 되풀이 하지 마시오.
                ### 참고 정보:{context} ### 사용자 질문 : {query}와 관련된 2~3가지 공약만 100자 이하로 답변해줘.  ###답변:"""

    # 응답 생성 - 토큰 수와 temperature 최적화
    answer = generate_answer(prompt, max_new_tokens=max_tokens, temperature=temperature)
    answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)
    return answer

# 기본 경로 테스트용
@app.get('/')
def hello_world():
    return '카카오톡 챗봇 LLM 서버가 실행 중입니다!'

# 카카오톡 스킬 엔드포인트
@app.post('/api/chat')
async def kakao_skill(request: Request):
    # 카카오톡 스킬 요청 파싱
    req = await request.json()

    # 사용자 발화(query) 추출
    user_query = req['userRequest']['utterance']

    # LLM으로 응답 생성
    answer = answer_with_rag(user_query)

    # 카카오톡 스킬 응답 형식으로 변환
    res = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "textCard": {
                        "title": f'🤖 {user_query} 관련 답변드립니다.',
                        "description": answer,
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
    uvicorn.run("chatbot-fastapi:app", host="0.0.0.0", port=5000)
