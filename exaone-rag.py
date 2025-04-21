import os
import torch
import time
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub.utils import logging

# 네트워크 연결 문제를 해결하기 위한 설정
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # 더 안정적인 전송 방식 활성화
os.environ["REQUESTS_CA_BUNDLE"] = ""  # SSL 인증서 문제 우회
os.environ["TRANSFORMERS_VERBOSITY"] = "info"  # 디버깅을 위한 자세한 로그

# 세션 타임아웃 증가
session = requests.Session()
session.mount('https://', requests.adapters.HTTPAdapter(
    max_retries=5,  # 재시도 횟수
    pool_connections=10,
    pool_maxsize=10,
    pool_block=True)
)
# 타임아웃 설정 (연결 타임아웃, 읽기 타임아웃)
session.request = lambda method, url, **kwargs: requests.Session.request(
    session, method, url, timeout=(30, 300), **kwargs)

logging.set_verbosity_warning()  # 로깅 레벨 조정
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# PDF 파일 경로 설정
pdf_paths = [
    "doc1.pdf",
    "doc2.pdf",
    "doc3.pdf"
]

# 1. PDF 문서 로드 및 텍스트 추출
documents = []
for pdf_path in pdf_paths:
    if os.path.exists(pdf_path):
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    else:
        print(f"파일이 존재하지 않습니다: {pdf_path}")

# 2. 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print(f"문서를 {len(chunks)}개의 청크로 분할했습니다.")

# 3. 임베딩 모델 설정
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/distiluse-base-multilingual-cased-v1",
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print(f"{model_name} 모델 로드 성공!")

# 4. 벡터 데이터베이스 생성
vectorstore = FAISS.from_documents(chunks, embedding_model)
print("벡터 데이터베이스 생성 완료")

# 5. EXAONE 모델 로드
model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
print(f"{model_name} 모델 로드 중... (큰 모델이므로 시간이 소요될 수 있습니다)")

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=False,  # 안정성을 위해 Fast Tokenizer 비활성화
    local_files_only=False  # 처음에는 온라인 시도
)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    revision="main",  # 명시적으로 메인 브랜치 사용
    low_cpu_mem_usage=True  # 메모리 사용량 최적화
)

# 6. LLM 파이프라인 설정
llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    top_p=0.95,
    repetition_penalty=1.15
)

# 7. RAG 파이프라인 설정
def exaone_predictor(prompt):
    system_prompt = "다음 질문에 대해 문서 정보를 바탕으로 자세히 답변해주세요."
    formatted_prompt = f"{system_prompt}\n\n{prompt}"
    
    response = llm(formatted_prompt)[0]['generated_text']
    # 입력 프롬프트를 제외한 실제 응답 부분만 추출
    actual_response = response[len(formatted_prompt):]
    return actual_response

# 8. RAG 검색 및 추론 기능 구현
def answer_question(query, k=3):
    # 관련 문서 검색
    relevant_docs = vectorstore.similarity_search(query, k=k)
    
    # 컨텍스트 생성
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # 프롬프트 구성
    prompt = f"""문맥:
{context}

질문: {query}

답변:"""
    
    # EXAONE 모델로 응답 생성
    answer = exaone_predictor(prompt)
    return answer

# 9. 질의응답 예시
def run_qa_example():
    while True:
        query = input("\n질문을 입력하세요 (종료하려면 'exit' 입력): ")
        if query.lower() == 'exit':
            break
        
        print("\n답변 생성 중...")
        answer = answer_question(query)
        print(f"\n답변: {answer}")

# 10. 실행
if __name__ == "__main__":
    print("PDF 문서 기반 EXAONE RAG 시스템이 준비되었습니다.")
    run_qa_example()
