from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import csv
import os
from datetime import datetime
import uuid

app = FastAPI()

# HTML 파일을 제공하기 위한 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=".")

# CSV 파일 경로 설정
CSV_FILE = "comments.csv"

# CSV 파일이 없으면 생성
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "text", "timestamp"])

# 댓글 모델 정의
class Comment(BaseModel):
    text: str

# 루트 경로에서 HTML 페이지 제공
@app.get("/", response_class=HTMLResponse)
async def get_html(request: Request):
    with open("index.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

# 댓글 목록 가져오기 API
@app.get("/api/comments")
async def get_comments():
    comments = []
    try:
        with open(CSV_FILE, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                comments.append({
                    "id": row["id"],
                    "text": row["text"],
                    "timestamp": row["timestamp"]
                })
        return {"success": True, "comments": comments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 댓글 저장하기 API
@app.post("/api/comments")
async def create_comment(comment: Comment):
    try:
        comment_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([comment_id, comment.text, timestamp])
        
        return {"success": True, "id": comment_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 서버 실행 방법
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
