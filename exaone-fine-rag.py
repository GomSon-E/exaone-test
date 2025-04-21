import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import fitz  # PyMuPDF
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import re
import time

# 전역 변수로 선언 (모든 함수에서 접근 가능)
tokenizer = None
retriever = None
document_store = None
index = None

# 1. PDF 파일에서 텍스트 추출하기
def extract_text_from_pdf(pdf_path):
    """PDF 파일에서 텍스트를 추출하는 함수"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# 2. 추출된 텍스트를 훈련 데이터로 변환
def prepare_training_data(pdf_files):
    """PDF 파일에서 텍스트를 추출하고 훈련 데이터로 변환"""
    texts = []
    
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            text = extract_text_from_pdf(pdf_file)
            # 긴 텍스트를 적절한 크기의 청크로 분할
            chunks = [text[i:i+512] for i in range(0, len(text), 384)]  # 오버랩을 위해 stride 적용
            texts.extend(chunks)
        else:
            print(f"파일을 찾을 수 없습니다: {pdf_file}")
    
    # 훈련 데이터 형식으로 변환
    training_data = []
    for text in texts:
        # 인스트럭션 형식으로 변환 (EXAONE 모델은 인스트럭션 튜닝을 지원)
        if len(text) > 200:
            input_text = text[:200]
            output_text = text[200:]
        else:
            input_text = text
            output_text = "텍스트에 대한 추가 정보가 없습니다."
            
        # 인스트럭션 형식으로 프롬프트 구성
        prompt = f"### Instruction:\n다음 텍스트에 대한 정보를 제공해주세요.\n\n### Input:\n{input_text}\n\n### Response:\n"
        
        training_example = {
            "text": prompt + output_text,
            "prompt_length": len(prompt)
        }
        training_data.append(training_example)
    
    return training_data

# 3. 데이터를 토크나이징하는 함수
def preprocess_function(examples):
    """데이터를 토크나이징하는 함수"""
    global tokenizer
    
    # 배치 처리 또는 단일 처리 모두 지원
    texts = examples["text"] if isinstance(examples["text"], list) else [examples["text"]]
    prompt_lengths = examples["prompt_length"] if isinstance(examples["prompt_length"], list) else [examples["prompt_length"]]
    
    # 토크나이징
    tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=512)
    
    result = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": [ids.copy() for ids in tokenized["input_ids"]]
    }
    
    # 프롬프트 부분에는 -100을 할당하여 loss 계산에서 제외
    for i, (ids, prompt_len) in enumerate(zip(result["labels"], prompt_lengths)):
        # 프롬프트 텍스트를 토큰화하여 정확한 토큰 길이 계산
        prompt_len_tokens = len(tokenizer(texts[i][:prompt_len], add_special_tokens=False)["input_ids"])
        
        # 특수 토큰 고려
        if tokenizer.bos_token_id is not None:
            prompt_len_tokens += 1  # BOS 토큰 추가로 인한 길이 조정
            
        # -100으로 마스킹
        if prompt_len_tokens < len(ids):
            result["labels"][i][:prompt_len_tokens] = [-100] * prompt_len_tokens
    
    return result

# 4. 모델 로드 및 LoRA 설정
def setup_model():
    """EXAONE 모델과 토크나이저를 로드하고 LoRA 설정"""
    global tokenizer  # 전역 변수 사용
    
    # 모델 및 토크나이저 로드
    model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    print(f"모델 로드 중: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("LoRA 설정 중...")
    # LoRA 설정 (메모리 효율적인 파인튜닝)
    lora_config = LoraConfig(
        r=16,  # LoRA의 랭크
        lora_alpha=32,  # LoRA의 스케일링 계수
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # LoRA 적용
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

# 5. 모델 파인튜닝 함수
def finetune_model(pdf_files, output_dir="./finetuned-exaone"):
    """모델을 파인튜닝하는 함수"""
    print("=" * 50)
    print("1단계: 모델 파인튜닝 시작")
    print("=" * 50)
    
    # 데이터 준비
    print("훈련 데이터 준비 중...")
    training_data = prepare_training_data(pdf_files)
    
    # 모델 설정
    model = setup_model()
    
    # 데이터셋 생성
    dataset = Dataset.from_list(training_data)
    
    # 디버깅: 데이터셋 확인
    print("데이터셋 샘플:", dataset[0])
    
    # 데이터셋 전처리 (토크나이징)
    print("토크나이징 중...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True
    )
    
    # 훈련 설정
    print("훈련 설정 중...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_steps=100,
        optim="adamw_torch",
    )
    
    # 트레이너 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # 훈련 실행
    print("훈련 시작...")
    trainer.train()
    
    # 모델 저장
    print(f"훈련된 모델 저장 중: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("파인튜닝 완료!")
    return model, output_dir

# 6. 파인튜닝된 모델 로드
def load_finetuned_model(model_path):
    """파인튜닝된 모델 로드"""
    global tokenizer
    
    print(f"파인튜닝된 모델 로드 중: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    return model

# 7. RAG 모델을 위한 문서 처리 함수
def prepare_rag_documents(pdf_files, chunk_size=512, chunk_overlap=128):
    """PDF 파일에서 텍스트를 추출하고 검색 가능한 문서 청크로 변환"""
    documents = []
    metadata = []
    
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            print(f"PDF 파일 처리 중: {pdf_file}")
            full_text = extract_text_from_pdf(pdf_file)
            
            # 텍스트를 더 작은 청크로 분할
            sentences = re.split(r'(?<=[.!?])\s+', full_text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                # 문장이 너무 길면 더 작게 분할
                if len(sentence) > chunk_size:
                    sentence_chunks = [sentence[i:i+chunk_size] for i in range(0, len(sentence), chunk_size-chunk_overlap)]
                    for sc in sentence_chunks:
                        if len(current_chunk) + len(sc) <= chunk_size:
                            current_chunk += sc + " "
                        else:
                            chunks.append(current_chunk.strip())
                            current_chunk = sc + " "
                else:
                    if len(current_chunk) + len(sentence) <= chunk_size:
                        current_chunk += sentence + " "
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # 각 청크에 메타데이터 추가
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadata.append({
                    "source": pdf_file,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                })
                
            print(f"  - {len(chunks)}개의 청크로 분할됨")
        else:
            print(f"파일을 찾을 수 없습니다: {pdf_file}")
    
    print(f"총 {len(documents)}개의 문서 청크가 준비되었습니다.")
    return documents, metadata

# 8. 임베딩 모델 초기화 및 벡터 데이터베이스 생성
def setup_rag_system(documents, metadata, embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """RAG 시스템 설정: 임베딩 모델 초기화 및 벡터 데이터베이스 생성"""
    global retriever, document_store, index
    
    print("임베딩 모델 로드 중...")
    # 임베딩 모델 초기화 (다국어 모델 사용)
    retriever = SentenceTransformer(embedding_model)
    print(f"임베딩 모델이 로드되었습니다: {embedding_model}")
    
    # 문서 임베딩 생성
    print("문서 임베딩 생성 중...")
    document_embeddings = retriever.encode(documents, show_progress_bar=True)
    
    # 문서 저장소 생성
    document_store = {
        "documents": documents,
        "metadata": metadata
    }
    
    # FAISS 인덱스 생성
    print("FAISS 인덱스 생성 중...")
    embedding_size = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_size)
    index.add(document_embeddings.astype(np.float32))
    
    print(f"FAISS 인덱스가 생성되었습니다. 총 {index.ntotal}개의 벡터가 인덱싱되었습니다.")
    
    # 데이터 저장 (나중에 재사용 가능)
    index_file = "rag_index.faiss"
    store_file = "rag_document_store.pkl"
    
    faiss.write_index(index, index_file)
    with open(store_file, "wb") as f:
        pickle.dump(document_store, f)
    
    print(f"RAG 데이터 저장됨: {index_file}, {store_file}")
    
    return index, document_store

# 9. RAG 시스템 설정
def setup_rag(pdf_files):
    """RAG 시스템 설정 함수"""
    print("=" * 50)
    print("2단계: RAG 시스템 설정 시작")
    print("=" * 50)
    
    # 문서 처리
    documents, metadata = prepare_rag_documents(pdf_files)
    
    # RAG 시스템 설정
    index, document_store = setup_rag_system(documents, metadata)
    
    print("RAG 시스템 설정 완료!")
    return index, document_store

# 10. 질의에 관련된 문서 검색
def retrieve_relevant_documents(query, top_k=3):
    """질의에 관련된 문서 검색"""
    global retriever, document_store, index
    
    # 질의 임베딩 생성
    query_embedding = retriever.encode([query])
    
    # FAISS로 유사한 문서 검색
    distances, indices = index.search(query_embedding.astype(np.float32), top_k)
    
    # 검색 결과 추출
    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx >= len(document_store["documents"]):
            continue  # 인덱스 범위 체크
        
        results.append({
            "content": document_store["documents"][idx],
            "metadata": document_store["metadata"][idx],
            "score": float(dist),
            "rank": rank + 1
        })
    
    return results

# 11. RAG 기반 추론 함수
def rag_inference(model, user_input, top_k=3, temperature=0.7):
    """RAG 기반 추론 함수: 관련 문서를 검색하고 이를 바탕으로 응답 생성"""
    global tokenizer
    
    # 시간 측정 시작
    start_time = time.time()
    
    # 관련 문서 검색
    retrieved_docs = retrieve_relevant_documents(user_input, top_k=top_k)
    
    # 검색된 문서 컨텍스트 준비 - 문서 길이 제한
    context = ""
    max_context_per_doc = 200  # 각 문서당 최대 문자 수 제한
    for i, doc in enumerate(retrieved_docs):
        doc_content = doc['content'].strip()
        if len(doc_content) > max_context_per_doc:
            doc_content = doc_content[:max_context_per_doc] + "..."
        context += f"[문서 {i+1}] {doc_content}\n\n"
    
    # 검색 시간 측정
    retrieval_time = time.time() - start_time
    
    # 인스트럭션 형식으로 변환 (검색 결과 포함)
    prompt = f"""### Instruction:
다음 정보를 바탕으로 사용자의 질문에 답변해주세요.

### Input:
사용자 질문: {user_input}

참고 문서:
{context}

### Response:
"""
    
    # 토크나이징 및 길이 확인
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 토큰 수 확인
    input_length = inputs["input_ids"].shape[1]
    print(f"입력 토큰 길이: {input_length}")
    
    # max_new_tokens 설정 (새로 생성할 최대 토큰 수)
    max_new_tokens = 512
    
    # 생성 설정
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,  # max_length 대신 max_new_tokens 사용
        "temperature": temperature,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    # 텍스트 생성 시간 측정 시작
    gen_start_time = time.time()
    
    # 텍스트 생성
    try:
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_kwargs)
        
        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # 프롬프트 부분 제거하고 응답만 반환
        response = generated_text[len(prompt):] if len(generated_text) > len(prompt) else generated_text
    except Exception as e:
        print(f"생성 중 오류 발생: {e}")
        # 오류 발생 시 간단한 응답 제공
        response = "죄송합니다, 응답 생성 중 오류가 발생했습니다. 질문을 더 짧게 해주시거나 다른 방식으로 질문해 주세요."
    
    # 생성 시간 측정
    generation_time = time.time() - gen_start_time
    total_time = time.time() - start_time
    
    # 응답과 함께 시간 정보 및 검색 결과 반환
    return {
        "response": response, 
        "retrieved_docs": retrieved_docs,
        "timing": {
            "retrieval": retrieval_time,
            "generation": generation_time,
            "total": total_time
        }
    }

# 12. RAG 기반 대화형 인터페이스
def interactive_rag_chat(model):
    """RAG 기반 대화형 채팅 인터페이스"""
    print("=" * 50)
    print("파인튜닝된 EXAONE 모델 + RAG 대화 시작")
    print("종료하려면 'exit' 또는 'quit'를 입력하세요.")
    print("=" * 50)
    
    while True:
        user_input = input("\n사용자: ")
        
        if user_input.lower() in ["exit", "quit", "종료"]:
            print("대화를 종료합니다.")
            break
        
        # RAG 기반 응답 생성
        result = rag_inference(model, user_input)
        response = result["response"]
        timing = result["timing"]
        retrieved_docs = result["retrieved_docs"]
        
        print(f"\nEXAONE: {response}")
        print(f"(응답 생성 시간: {timing['total']:.2f}초, 검색: {timing['retrieval']:.2f}초, 생성: {timing['generation']:.2f}초)")
        
        # 검색된 문서 정보 표시
        if retrieved_docs:
            print("\n참고한 문서:")
            for i, doc in enumerate(retrieved_docs[:2]):  # 상위 2개 문서만 표시
                source = doc["metadata"]["source"]
                print(f"- 문서 {i+1}: {os.path.basename(source)} (유사도: {1 - doc['score']:.2f})")

# 13. 메인 함수: 파인튜닝 후 RAG 적용
def main(pdf_files, output_dir="./finetuned-exaone"):
    """메인 함수: 모델 파인튜닝 후 RAG 적용"""
    # 1. 모델 파인튜닝
    model, model_path = finetune_model(pdf_files, output_dir)
    
    # 2. RAG 시스템 설정
    index, document_store = setup_rag(pdf_files)
    
    # 3. RAG 기반 대화형 인터페이스 실행
    interactive_rag_chat(model)

# 실행 코드
if __name__ == "__main__":
    # PDF 파일 경로
    pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    
    # 메인 함수 실행 (파인튜닝 후 RAG 적용)
    main(pdf_files)
