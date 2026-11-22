import io
import os
from datetime import date
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True, encoding="utf-8-sig")

api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
client = genai.Client(api_key=api_key)

# 요청한 모델 우선 시도 후, 응답 불가/지연 시 안정적인 flash 모델로 자동 폴백
CANDIDATE_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]

app = FastAPI()


class ReceiptFeedback(BaseModel):
    date: str = Field(description="결제일 YYYY-MM-DD (없으면 오늘 날짜)")
    merchant: str = Field(description="상호명 또는 대표 메뉴명")
    amount: int = Field(description="할인 반영 후 최종 결제 금액(원 단위 정수)")
    category: Literal["식비", "배달", "카페/간식", "교통", "쇼핑/생활", "기타"]
    score: int = Field(description="소비 합리성 점수 (0~100)")
    stamp: str = Field(description="도장 문구 2~4글자 (예: 합격, 주의, 과소비, 알뜰)")
    badge: str = Field(description="소비 캐릭터 한 줄 별명")
    headline: str = Field(description="핵심 한 줄 팩트폭행/칭찬 코멘트")
    good_point: str = Field(description="이 소비에서 잘한 점 1가지")
    bad_point: str = Field(description="아쉬운 점 또는 주의할 점 1가지")
    next_mission: str = Field(description="다음 소비 때 바로 써먹을 현실적인 절약 미션 1가지")


PROMPT = f"""
당신은 감각적이고 현실적인 소비 피드백 앱의 AI 코치입니다.
오늘 날짜는 {date.today().isoformat()} 입니다.
첨부된 영수증 또는 배달앱 결제 캡처 이미지를 빠르게 파악해 JSON으로만 답하세요.
- 배달앱 캡처는 쿠폰/할인 적용 후 '최종 결제 금액'을 기준으로 계산하세요.
- 길고 군더더기 없는 문장으로 간결하고 재치 있게 작성하세요.
"""


def prepare_image(raw_bytes: bytes) -> Image.Image:
    """긴 캡처나 고해상도 이미지를 빠르게 분석할 수 있도록 최적화합니다."""
    img = Image.open(io.BytesIO(raw_bytes))
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    # 1024px 수준으로 줄이면 글자 인식률은 유지하면서 전송·추론 속도가 빨라집니다.
    img.thumbnail((1024, 1024))
    return img


@app.get("/", response_class=HTMLResponse)
async def home():
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not api_key:
        raise HTTPException(status_code=500, detail=".env 파일에 GEMINI_API_KEY가 없습니다.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="빈 이미지 파일입니다.")

    img = prepare_image(raw_bytes)
    last_error = ""

    for model_name in CANDIDATE_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[img, PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ReceiptFeedback,
                    temperature=0.3,
                ),
            )
            return ReceiptFeedback.model_validate_json(response.text)
        except Exception as e:
            last_error = f"{model_name}: {str(e)}"
            continue

    raise HTTPException(status_code=500, detail=f"분석 실패 -> {last_error}")