# main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
import pandas as pd
from io import StringIO
import json
import time
from typing import Dict, List, Optional
import asyncio
from datetime import datetime
import os

app = FastAPI(title="구글 폼 응답 API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시에는 특정 도메인으로 제한하세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory="."), name="static")

# 캐시를 위한 전역 변수
responses_cache = {
    "data": [],
    "last_updated": None,
    "is_updating": False
}

# 구글 시트 URL과 ID 설정
SHEET_ID = "2PACX-1vSwlXn0qXASDxBm2QDZ9BcSlWMi01k-qhM6bBACHrMOpjjV5-WtOpB7eu2hZaZANraTFJx2Kf-hOct2"  # 구글 시트 ID 입력
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

async def fetch_google_sheet_data():
    """구글 시트에서 데이터 가져오기"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(CSV_URL, timeout=30.0)
            
            if response.status_code != 200:
                return {"error": f"구글 시트에서 데이터를 가져올 수 없습니다. 상태 코드: {response.status_code}"}
            
            # CSV 데이터 파싱
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            
            # 데이터프레임을 딕셔너리 목록으로 변환
            return json.loads(df.to_json(orient="records", force_ascii=False))
    except Exception as e:
        return {"error": f"데이터 가져오기 실패: {str(e)}"}

async def update_cache_background():
    """백그라운드에서 캐시 업데이트"""
    try:
        responses_cache["is_updating"] = True
        data = await fetch_google_sheet_data()
        
        if "error" not in data:
            responses_cache["data"] = data
            responses_cache["last_updated"] = datetime.now().isoformat()
        
        responses_cache["is_updating"] = False
    except Exception as e:
        responses_cache["is_updating"] = False
        print(f"캐시 업데이트 중 오류 발생: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 초기 데이터 로드"""
    await update_cache_background()

@app.get("/api/responses")
async def get_responses(background_tasks: BackgroundTasks, force_refresh: bool = False):
    """구글 폼 응답 데이터 반환"""
    # 강제 새로고침 또는 캐시가 비어있는 경우 데이터 가져오기
    if force_refresh or not responses_cache["data"]:
        if not responses_cache["is_updating"]:
            background_tasks.add_task(update_cache_background)
    
    # 캐시된 데이터와 마지막 업데이트 시간 반환
    return {
        "data": responses_cache["data"],
        "last_updated": responses_cache["last_updated"],
        "is_updating": responses_cache["is_updating"]
    }

@app.get("/api/stats")
async def get_stats():
    """기본 통계 데이터 계산 및 반환"""
    data = responses_cache["data"]
    
    if not data:
        return {"error": "데이터가 없습니다"}
    
    try:
        # 타임스탬프 필드 찾기 (일반적으로 'Timestamp' 또는 '타임스탬프')
        timestamp_field = next((field for field in data[0].keys() 
                               if field.lower() in ['timestamp', '타임스탬프']), None)
        
        # 오늘 날짜의 응답 수 계산
        today = datetime.now().date()
        today_responses = 0
        
        if timestamp_field:
            for entry in data:
                try:
                    entry_date = datetime.fromisoformat(entry[timestamp_field].replace('Z', '+00:00')).date()
                    if entry_date == today:
                        today_responses += 1
                except:
                    pass
        
        return {
            "total_responses": len(data),
            "today_responses": today_responses,
            "fields": list(data[0].keys()) if data else []
        }
    except Exception as e:
        return {"error": f"통계 계산 중 오류 발생: {str(e)}"}

# 특정 간격으로 백그라운드 업데이트 실행 (선택사항)
@app.on_event("startup")
async def schedule_periodic_updates():
    """일정 간격으로 데이터 자동 업데이트"""
    async def periodic_update():
        while True:
            await asyncio.sleep(300)  # 5분마다 업데이트
            if not responses_cache["is_updating"]:
                await update_cache_background()
    
    # 백그라운드 태스크로 실행
    asyncio.create_task(periodic_update())

# 홈페이지 라우트 - index.html 반환
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# app.js 파일 제공
@app.get("/app.js")
async def get_app_js():
    return FileResponse("app.js", media_type="application/javascript")

# 상태 체크용 엔드포인트
@app.get("/status")
async def status():
    return {"status": "online", "service": "구글 폼 응답 API"}

# 애플리케이션 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=True)
