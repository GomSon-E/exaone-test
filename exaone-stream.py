import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import sys

# 모델과 토크나이저 로드
print("모델을 로딩 중입니다... 잠시 기다려주세요.")
model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
print("모델 로딩 완료!")

# 응답 생성
def generate_answer(question):
    # 입력 인코딩
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    attention_mask = inputs.attention_mask

    max_new_tokens = 200

    # 토큰 생성 및 스트리밍 출력
    print("\n응답: ", end="")

    # 첫 번째 토큰 생성
    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    prev_input_len = inputs.input_ids.shape[1]
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
                temperature=0.7,
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

    print("\n\n생성 완료")

# 사용자 입력 받기
question = input("\n질문을 입력하세요: ")

generate_answer(question)

# 계속 대화하기
while True:
    choice = input("\n계속 질문하시겠습니까? (y/n): ")
    if choice.lower() != 'y':
        print("프로그램을 종료합니다.")
        break

    question = input("\n질문을 입력하세요: ")

    generate_answer(question)
