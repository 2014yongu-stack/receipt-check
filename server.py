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
    "교통": ["버스", "지하철", "택시", "주유", "코레일", "이체", "송금"],
    "쇼핑/생활": ["마트", "편의점", "cu", "gs25", "세븐일레븐", "다이소", "올리브영", "은행", "뱅크", "토스", "페이"],
    "기타": [],
}

app = FastAPI()


class ReceiptFeedback(BaseModel):
    date: str = Field(description="결제일 또는 이체일 YYYY-MM-DD (없으면 오늘 날짜)")
    source_type: Literal["영수증", "배달앱", "은행앱", "복합내역"] = Field(
        description="캡처 출처 구분 (은행 이체/출금 화면이면 '은행앱', 함께 올렸으면 '복합내역')"
    )
    merchant: str = Field(description="상호명 또는 수취인/은행 거래처명")
    items_summary: str = Field(
        description="가계부에 기록할 핵심 품목 또는 이체·출금 내역 요약 (여러 장이면 누락 없이 함께 기재)"
    )
    amount: int = Field(description="할인 반영 후 최종 결제 금액 또는 출금 금액(원 단위 정수)")
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


def build_prompt(image_count: int = 1, target_date: str | None = None, style: str = "실용적이고 재치 있게", **kwargs: Any) -> str:
    """기본/키워드 매개변수, format(), 날짜·시간 활용"""
    now_dt = datetime.now()
    today_str = target_date if target_date else date.today().isoformat()
    time_str = now_dt.strftime("%H:%M:%S")
    tone = str(kwargs.get("tone", "현실적")).strip()

    multi_note = (
        f"첨부된 {image_count}장의 이미지를 모두 빠짐없이 읽으세요.\n"
        "- 같은 주문이 길어서 나뉜 캡처라면 중복 합산하지 말고 하나로 통합하세요.\n"
        "- 만약 '은행 앱(이체/출금/결제 내역)'과 '배달앱/영수증' 등 서로 다른 내역이 함께 올라왔다면 어느 한쪽을 임의로 배제하거나 무시하지 말고, 은행 앱의 거래처·이체/출금 금액과 배달앱/영수증 내역을 모두 items_summary와 headline/bad_point에 함께 반영하세요."
        if image_count > 1
        else "첨부된 1장의 이미지(영수증, 배달앱 결제 캡처, 또는 은행 앱 이체·결제·출금 화면)를 분석하세요. 은행 앱 화면도 유효한 지출/거래 내역으로 적극 인식하세요."
    )

    template = (
        "당신은 영수증, 배달앱 캡처, 은행 앱 거래내역을 모두 읽고 가계부를 자동 정리하는 AI 자산관리사입니다.\n"
        "현재 시각은 {today} {now_time} 입니다.\n"
        "{multi_note}\n"
        "- 은행 앱 화면에서는 출금/이체/결제 금액, 상대방/거래처명, 날짜를 정확히 추출하세요.\n"
        "- 배달앱 캡처는 쿠폰/할인 적용 후 '최종 결제 금액'을 기준으로 계산하세요.\n"
        "- stamp는 반드시 '알뜰', '합격', '주의', '과소비' 중 하나로 답하세요.\n"
        "- 문체는 '{style}'하고 '{tone}'인 톤으로 작성하세요."
    )
    return template.format(
        tone=tone,
        today=today_str,
        now_time=time_str,
        multi_note=multi_note,
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
    img.thumbnail((1600, 1600))
    return img


@app.get("/", response_class=HTMLResponse)
async def home():
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.post("/analyze")
async def analyze(files: list[UploadFile] = File(...)):
    if not api_key:
        raise HTTPException(status_code=500, detail=".env 파일에 GEMINI_API_KEY가 없습니다.")

    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="업로드된 이미지가 없습니다.")

    images: list[Image.Image] = []
    for f in files:
        raw_bytes = await f.read()
        if raw_bytes:
            images.append(prepare_image(raw_bytes))

    if len(images) == 0:
        raise HTTPException(status_code=400, detail="유효한 이미지 파일이 없습니다.")

    prompt_text = build_prompt(image_count=len(images), style="실용적이고 재치 있게", tone="현실적")
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
                contents=[*images, prompt_text],
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