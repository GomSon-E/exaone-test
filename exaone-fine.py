import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import fitz  # PyMuPDF
from peft import LoraConfig, get_peft_model

# 전역 변수로 토크나이저 선언 (모든 함수에서 접근 가능)
tokenizer = None

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
    model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
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

# 5. 메인 훈련 함수
def train_model(pdf_files, output_dir="./finetuned-exaone"):
    """모델 파인튜닝 실행 함수"""
    # 데이터 준비
    training_data = prepare_training_data(pdf_files)
    
    # 모델 설정
    model = setup_model()
    
    # 데이터셋 생성
    dataset = Dataset.from_list(training_data)
    
    # 디버깅: 데이터셋 확인
    print("데이터셋 샘플:", dataset[0])
    
    # 데이터셋 전처리 (토크나이징)
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True
    )
    
    # 훈련 설정
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
    trainer.train()
    
    # 모델 저장
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model

# 6. 대화형 인터페이스를 위한 추론 함수
def inference(model, user_input, max_length=512):
    """사용자 입력에 대한 응답 생성"""
    global tokenizer
    
    # 인스트럭션 형식으로 변환
    prompt = f"### Instruction:\n사용자의 질문에 답변해주세요.\n\n### Input:\n{user_input}\n\n### Response:\n"
    
    # 토크나이징
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 생성 설정
    gen_kwargs = {
        "max_length": max_length,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    # 텍스트 생성
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    
    # 생성된 텍스트 디코딩
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # 프롬프트 부분 제거하고 응답만 반환
    response = generated_text[len(prompt):]
    
    return response

# 7. 대화형 인터페이스
def interactive_chat(model):
    """대화형 채팅 인터페이스"""
    print("파인튜닝된 EXAONE 모델과 대화를 시작합니다. 종료하려면 'exit' 또는 'quit'를 입력하세요.")
    
    while True:
        user_input = input("\n사용자: ")
        
        if user_input.lower() in ["exit", "quit", "종료"]:
            print("대화를 종료합니다.")
            break
        
        response = inference(model, user_input)
        print(f"\nEXAONE: {response}")

# 메인 실행 코드
if __name__ == "__main__":
    # PDF 파일 경로
    pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    
    # 모델 파인튜닝
    print("모델 파인튜닝을 시작합니다...")
    model = train_model(pdf_files)
    
    # 대화형 인터페이스 실행
    interactive_chat(model)
