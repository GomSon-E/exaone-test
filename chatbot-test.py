from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    return '카카오톡 스킬 테스트 서버가 실행 중입니다!'

@app.route('/api/chat', methods=['POST'])
def kakao_skill():
    try:
        # 카카오톡 요청 데이터 파싱
        req = request.get_json()
        
        # 사용자 발화 추출
        user_query = req['userRequest']['utterance']
        
        # 사용자 발화의 앞 5글자만 추출 (5글자보다 짧을 경우 전체 반환)
        if len(user_query) > 5:
            answer = user_query[:5] + "..."
        else:
            answer = user_query
            
        # 간단한 설명 추가
        response_text = f"입력하신 메시지의 앞부분: '{answer}'\n(테스트 응답입니다)"
        
        # 카카오톡 응답 형식으로 변환
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": response_text
                        }
                    }
                ]
            }
        }
        
        return jsonify(res)
        
    except Exception as e:
        # 오류 발생 시 디버깅용 응답
        error_response = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": f"오류가 발생했습니다: {str(e)}"
                        }
                    }
                ]
            }
        }
        return jsonify(error_response)

if __name__ == "__main__":
    # 외부에서 접속 가능하도록 호스트 설정
    app.run(host='0.0.0.0', port=5000, debug=True)
