import os
import torch
import sys
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# PDF 파일 경로 설정
pdf_paths = [
    "doc1.pdf",
    "doc2.pdf"
]

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
    exit(1)

# 2. 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
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

# 5. EXAONE 모델 로드
model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
print(f"{model_name} 모델 로드 중... (큰 모델이므로 시간이 소요될 수 있습니다)")

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 모델 로드 - 성능 최적화 설정
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 정밀도 낮추기 (속도 향상)
    device_map="auto",          # 자동 장치 할당
    trust_remote_code=True,
    low_cpu_mem_usage=True      # CPU 메모리 사용량 최적화
)
print("EXAONE 모델 로드 완료!")

# 6. 커스텀 TextStreamer 클래스 정의 - 수정된 부분
class FastTextStreamer(TextStreamer):
    """최적화된 텍스트 스트리밍 클래스"""

    def __init__(self, tokenizer, skip_prompt=False, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **decode_kwargs)
        self.prefix = "\n답변: "
        self.printed_prefix = False

    def on_finalized_text(self, text, stream_end=False):
        if not self.printed_prefix:
            print(self.prefix, end="")
            self.printed_prefix = True

        # 생성된 텍스트 출력
        print(text, end="", flush=True)

# 7. RAG 검색 함수
def retrieve_context(query, k=3):
    """쿼리와 관련된 문서를 검색하여 컨텍스트를 생성합니다."""
    relevant_docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    return context

# 8. 최적화된 스트리밍 응답 생성 함수
def generate_streaming_answer(prompt, max_new_tokens=200, temperature=0.7):
    """프롬프트에 대한 응답을 최적화된 스트리밍 방식으로 생성합니다."""
    # 입력 인코딩
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 스트리머 준비
    streamer = FastTextStreamer(tokenizer, skip_prompt=True)

    # 효율적인 생성 설정
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True if temperature > 0 else False,
        "temperature": temperature,
        "top_p": 0.92,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # 토큰 생성 (자동 스트리밍)
    with torch.no_grad():
        model.generate(
            inputs.input_ids,
            streamer=streamer,
            **generation_config
        )

    print("\n")
    return

# 9. RAG + 스트리밍 통합 함수
def answer_with_rag_streaming(query, k=3, max_tokens=300, temperature=0.5):
    """RAG로 컨텍스트를 검색하고 스트리밍 방식으로 응답을 생성합니다."""
    print("관련 문서를 검색 중...")
    context = retrieve_context(query, k=k)

    # 프롬프트 구성
    system_prompt = "다음 질문에 대해 제공된 문서 정보를 바탕으로 자세히 답변해주세요."
    prompt = f"""{system_prompt}

문맥:
{context}

질문: {query}

답변:"""

    # 스트리밍 응답 생성
    generate_streaming_answer(prompt, max_new_tokens=max_tokens, temperature=temperature)

# 10. 실행 함수
def run_rag_streaming():
    print("PDF 문서 기반 EXAONE RAG 스트리밍 시스템이 준비되었습니다!")
    print("--------------------------------------------------------------")

    while True:
        query = input("\n질문을 입력하세요 (종료하려면 'exit' 입력): ")
        if query.lower() == 'exit':
            print("프로그램을 종료합니다.")
            break

        answer_with_rag_streaming(query)

# 11. 성능 최적화 설정
def optimize_performance():
    """성능 최적화 설정을 적용합니다."""
    # GPU 메모리 최적화
    if torch.cuda.is_available():
        # 사용 가능한 가장 빠른 CUDA 알고리즘 사용
        torch.backends.cudnn.benchmark = True

        # 메모리 캐싱 최적화
        torch.cuda.empty_cache()

        # 비동기 CUDA 연산 활성화
        torch.cuda.set_device(torch.cuda.current_device())

    # 모델 최적화
    if hasattr(model, 'config'):
        # 캐싱 최적화
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = True

        # 병렬 처리 최적화
        if hasattr(model.config, 'gradient_checkpointing'):
            model.config.gradient_checkpointing = False

    print("성능 최적화 설정이 적용되었습니다.")

# 12. 실행
if __name__ == "__main__":
    # 성능 최적화 설정 적용
    optimize_performance()

    # RAG 스트리밍 시스템 실행
    run_rag_streaming()
