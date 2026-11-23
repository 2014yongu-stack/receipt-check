from datetime import date, datetime
import io
import os
from pathlib import Path
from typing import Any, Iterator, Literal

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

CATEGORIES_TUPLE: tuple[str, ...] = ("식비", "배달", "카페/간식", "교통", "쇼핑/생활", "기타")
DEFAULT_MODELS: tuple[str, ...] = ("gemini-3.5-flash-lite", "gemini-3.5-flash")

CANDIDATE_MODELS: list[str] = list(DEFAULT_MODELS)
CANDIDATE_MODELS.append("gemini-2.5-flash")
CANDIDATE_MODELS.insert(0, "gemini-3.5-flash-lite")

while len(CANDIDATE_MODELS) > len(set(CANDIDATE_MODELS)):
    seen: list[str] = []
    for idx, name in enumerate(CANDIDATE_MODELS):
        if name in seen:
            CANDIDATE_MODELS.pop(idx)
            break
        seen.append(name)

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "배달": ["배민", "배달", "쿠팡이츠", "요기요"],
    "카페/간식": ["커피", "카페", "스타벅스", "메가", "컴포즈", "마카롱", "베이커리"],
    "식비": ["식당", "초밥", "우동", "밥", "정식", "국밥", "김밥"],
    "교통": ["버스", "지하철", "택시", "주유", "코레일"],
    "쇼핑/생활": ["마트", "편의점", "cu", "gs25", "세븐일레븐", "다이소", "올리브영"],
    "기타": [],
}

app = FastAPI()


class ReceiptFeedback(BaseModel):
    date: str = Field(description="결제일 YYYY-MM-DD (없으면 오늘 날짜)")
    merchant: str = Field(description="상호명")
    items_summary: str = Field(description="가계부에 기록할 핵심 구매 품목 요약 (예: 아메리카노 외 1건)")
    amount: int = Field(description="할인 반영 후 최종 결제 금액(원 단위 정수)")
    category: Literal["식비", "배달", "카페/간식", "교통", "쇼핑/생활", "기타"]
    score: int = Field(description="소비 합리성 점수 (0~100)")
    stamp: str = Field(description="판정 도장 2~4글자 (알뜰, 합격, 주의, 과소비 중 택1)")
    badge: str = Field(description="이번 소비에 어울리는 재치 있는 한 줄 별명")
    headline: str = Field(description="가계부 한 줄 요약 및 소비 습관 코멘트")
    good_point: str = Field(description="이 소비에서 잘한 점 1가지")
    bad_point: str = Field(description="아쉬운 점 또는 절약 포인트 1가지")
    next_mission: str = Field(description="다음 소비 때 바로 실천할 현실적 절약 미션 1가지")


def clean_text(raw_text: str, *, mode: str = "keep") -> str:
    """조기리턴, strip(), upper(), lower(), find(), rfind(), split(), join() 활용"""
    if not raw_text:
        return ""

    trimmed = raw_text.strip()
    if trimmed == "":
        return ""

    left_idx = trimmed.find("(")
    right_idx = trimmed.rfind(")")
    if left_idx != -1 and right_idx > left_idx:
        trimmed = (trimmed[:left_idx] + trimmed[right_idx + 1 :]).strip()

    words = [w.strip() for w in trimmed.split() if w.strip()]
    joined = " ".join(words)

    if mode.lower() == "upper":
        return joined.upper()
    elif mode.lower() == "lower":
        return joined.lower()
    else:
        return joined


def classify_merchant(merchant: str, *extra_tags: str, default_cat: str = "기타") -> str:
    """가변매개변수, 기본 매개변수, in, not/and/or, if~elif~else, 딕셔너리 반복문"""
    target = clean_text(merchant, mode="lower")
    if target == "" and len(extra_tags) == 0:
        return default_cat

    normalized_tags: list[str] = []
    for tag in extra_tags:
        cleaned = clean_text(tag, mode="lower")
        if cleaned != "":
            normalized_tags.append(cleaned)

    for cat_name, keywords in CATEGORY_KEYWORDS.items():
        if cat_name == "기타":
            continue
        for kw in keywords:
            if (kw in target) or (kw in normalized_tags):
                return cat_name

    if ("할인" in target or "쿠폰" in target) and not ("배달" in target):
        return "쇼핑/생활"
    elif default_cat in CATEGORIES_TUPLE:
        return default_cat
    else:
        return "기타"


def build_prompt(target_date: str | None = None, style: str = "실용적이고 재치 있게", **kwargs: Any) -> str:
    """기본/키워드 매개변수, format(), 날짜·시간 활용"""
    now_dt = datetime.now()
    today_str = target_date if target_date else date.today().isoformat()
    time_str = now_dt.strftime("%H:%M:%S")
    tone = str(kwargs.get("tone", "현실적")).strip()

    template = (
        "당신은 영수증 한 장으로 사용자의 가계부를 자동 정리하고 소비 습관을 코칭하는 AI 자산관리사입니다.\n"
        "현재 시각은 {today} {now_time} 입니다.\n"
        "첨부된 영수증 또는 배달앱 결제 캡처 이미지를 분석해 JSON으로만 답하세요.\n"
        "- 배달앱 캡처는 쿠폰/할인 적용 후 '최종 결제 금액'을 기준으로 계산하세요.\n"
        "- items_summary에는 가계부 내역에 바로 적힐 깔끔한 품목 요약을 넣으세요.\n"
        "- stamp는 반드시 '알뜰', '합격', '주의', '과소비' 중 하나로 답하세요.\n"
        "- 문체는 '{style}'하고 '{tone}'인 톤으로 작성하세요."
    )
    return template.format(
        tone=tone,
        today=today_str,
        now_time=time_str,
        style=style.lower().strip(),
    )


def countdown_generator(n: int) -> Iterator[int]:
    """제너레이터(yield)와 재귀 함수"""
    if n <= 0:
        return
    yield n
    yield from countdown_generator(n - 1)


def prepare_image(raw_bytes: bytes) -> Image.Image:
    """고해상도 이미지 최적화"""
    img = Image.open(io.BytesIO(raw_bytes))
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
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
    prompt_text = build_prompt(style="실용적이고 재치 있게", tone="현실적")
    last_error = ""

    model_iter: Iterator[str] = iter(CANDIDATE_MODELS)
    tried: list[str] = []

    while True:
        try:
            model_name = next(model_iter)
        except StopIteration:
            break

        if not model_name or model_name in tried:
            continue

        tried.append(model_name)

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[img, prompt_text],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ReceiptFeedback,
                    temperature=0.3,
                ),
            )
            parsed = ReceiptFeedback.model_validate_json(response.text)
            parsed.merchant = clean_text(parsed.merchant, mode="keep") or "미확인 상호"
            if not parsed.category:
                pass
            return parsed
        except Exception as e:
            last_error = f"{model_name}: {str(e)}"
            continue

    raise HTTPException(status_code=500, detail=f"분석 실패 -> {last_error}")


def run_cli_demo() -> None:
    """터미널에서 python server.py 로 직접 실행할 때 동작하는 CLI 실습 블록"""
    user_input = input("테스트할 상호명을 입력하세요 (엔터 시 샘플 실행): ")
    records: list[dict[str, Any]] = [
        {"merchant": "학교 앞 식당", "amount": 8000, "score": 90},
        {"merchant": "동네 카페", "amount": 4500, "score": 85},
    ]

    if clean_text(user_input) != "":
        guessed = classify_merchant(user_input, "할인", "쿠폰")
        records.append({"merchant": clean_text(user_input), "amount": 12000, "score": 75, "category": guessed})

    records.sort(key=lambda r: r["score"])
    for rank, item in enumerate(reversed(records), start=1):
        print("순위 {}위: {} ({:,}원)".format(rank, item.get("merchant", "-"), item.get("amount", 0)))

    temp = ["a", "b", "c"]
    del temp[0]
    temp.remove("b")
    temp.clear()

    for tick in countdown_generator(3):
        print("카운트:", tick)


if __name__ == "__main__":
    run_cli_demo()