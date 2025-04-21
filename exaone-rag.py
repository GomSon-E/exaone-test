import os
import torch
import sys
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# PDF 파일 경로 설정
pdf_paths = [
    "doc1.pdf",
    "doc2.pdf",
    "doc3.pdf"
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
model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
print(f"{model_name} 모델 로드 중... (큰 모델이므로 시간이 소요될 수 있습니다)")

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
print("EXAONE 모델 로드 완료!")

# 6. RAG 검색 함수
def retrieve_context(query, k=3):
    """쿼리와 관련된 문서를 검색하여 컨텍스트를 생성합니다."""
    relevant_docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    return context

# 7. 스트리밍 응답 생성 함수
def generate_streaming_answer(prompt, max_new_tokens=200, temperature=0.7):
    """프롬프트에 대한 응답을 스트리밍 방식으로 생성합니다."""
    # 입력 인코딩
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    attention_mask = inputs.attention_mask
    prev_input_len = inputs.input_ids.shape[1]
    
    print("\n답변: ", end="")
    sys.stdout.flush()
    
    # 첫 번째 토큰 생성
    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(generated_ids[0][prev_input_len:], skip_special_tokens=True)
    sys.stdout.write(generated_text)
    sys.stdout.flush()
    
    # 나머지 토큰 생성
    for i in range(max_new_tokens - 1):
        # 이전 출력을 새 입력으로 사용
        with torch.no_grad():
            next_token_ids = model.generate(
                generated_ids,
                max_new_tokens=1,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 마지막에 생성된 토큰만 가져오기
        next_token = next_token_ids[0][-1].unsqueeze(0)
        
        # 토큰 디코딩
        next_token_text = tokenizer.decode(next_token, skip_special_tokens=True)
        
        # 생성된 텍스트 출력
        sys.stdout.write(next_token_text)
        sys.stdout.flush()
        
        # 생성된 ID 업데이트
        generated_ids = next_token_ids
        
        # EOS 토큰이 생성되면 중단
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    print("\n")
    return

# 8. RAG + 스트리밍 통합 함수
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
    
    print("답변 생성 중...")
    generate_streaming_answer(prompt, max_new_tokens=max_tokens, temperature=temperature)

# 9. 실행 함수
def run_rag_streaming():
    print("PDF 문서 기반 EXAONE RAG 스트리밍 시스템이 준비되었습니다!")
    print("--------------------------------------------------------------")
    
    while True:
        query = input("\n질문을 입력하세요 (종료하려면 'exit' 입력): ")
        if query.lower() == 'exit':
            print("프로그램을 종료합니다.")
            break
        
        answer_with_rag_streaming(query)

# 10. 실행
if __name__ == "__main__":
    run_rag_streaming()
