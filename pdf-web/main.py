from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path

app = FastAPI()

# CORS 설정 - 프론트엔드 도메인을 추가하세요
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중에는 모든 오리진 허용, 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 디렉토리 설정
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
FILES_DIR = BASE_DIR / "files"

# 정적 파일 디렉토리가 없으면 생성
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(FILES_DIR, exist_ok=True)

# 정적 파일 제공 설정 (HTML, CSS, JS 파일)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/files")
async def get_files():
    """PDF 파일 목록을 반환하는 API"""
    files = {
        "full": {
            "path": "/api/download/full_document.pdf",
            "thumbnail": "/api/thumbnail/full_document_thumbnail.jpg",
            "title": "통합 문서"
        }
    }
    
    # 개별 파일 8개 추가
    for i in range(1, 9):
        files[f"file{i}"] = {
            "path": f"/api/download/file{i}.pdf",
            "title": f"{i}번째 파일"
        }
    
    return files

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """파일 다운로드 API"""
    file_path = FILES_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # 파일 다운로드 응답 생성
    # filename 파라미터는 브라우저가 다운로드할 때 표시할 파일 이름
    return FileResponse(
        path=file_path, 
        filename=filename,
        media_type="application/pdf"
    )

@app.get("/api/thumbnail/{filename}")
async def get_thumbnail(filename: str):
    """썸네일 이미지 제공 API"""
    file_path = FILES_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    
    return FileResponse(
        path=file_path,
        media_type="image/jpeg"  # 썸네일 형식에 맞게 조정
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=True)
