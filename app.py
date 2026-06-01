import json
import math
import tempfile
import threading
import time
import uuid
import re
import html as html_module
from collections import defaultdict
from urllib.parse import quote, unquote
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel
from typing import Optional, Literal
import gradio as gr

# ============================================================
# Config
# ============================================================
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

MAX_SESSIONS = 2
MAX_MESSAGES_PER_SESSION = 10

MISSING_MESSAGES = {
    "destination": "어디로 여행을 가고 싶으신지 알려주시면 계획을 짜드리겠습니다! 😊",
    "duration":    "여행 기간은 어느 정도로 생각하고 계신가요? 😊",
    "both":        "어디로, 얼마 동안 여행을 가고 싶으신지 알려주시면 계획을 짜드리겠습니다! 😊",
}

CHITCHAT_RESPONSE = (
    "이 챗봇은 여행 관련된 응답만 제공 가능합니다. "
    "여행지와 기간을 알려주시면 계획을 짜드릴게요! ✈️"
)

# Day-color palette for map markers (wraps after 7 days)
DAY_COLORS = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue"]

MAP_EMPTY_HTML = (
    "<div style='height:300px;display:flex;align-items:center;"
    "justify-content:center;color:#9ca3af;font-size:14px;'>"
    "🗺️ 일정을 생성하면 경로 지도가 표시됩니다</div>"
)
MAP_LOADING_HTML = (
    "<div style='height:300px;display:flex;flex-direction:column;align-items:center;"
    "justify-content:center;gap:12px;'>"
    "<div style='font-size:28px;animation:spin 1s linear infinite;'>🗺️</div>"
    "<p style='color:#6b7280;font-size:13px;margin:0;'>"
    "지도 좌표를 가져오는 중입니다…</p>"
    "<style>@keyframes spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}</style>"
    "</div>"
)
MAP_ERROR_HTML = (
    "<div style='height:300px;display:flex;align-items:center;"
    "justify-content:center;color:#9ca3af;font-size:13px;'>"
    "지도를 표시하지 못했습니다. 각 장소의 Google Maps 링크를 이용해주세요.</div>"
)

# Korean city → (English name, Country, center_lat, center_lng, max_radius_km)
DEST_EN_MAP: dict[str, tuple] = {
    "도쿄":       ("Tokyo",           "Japan",       35.6762,  139.6503,  80),
    "오사카":     ("Osaka",           "Japan",       34.6937,  135.5023,  60),
    "교토":       ("Kyoto",           "Japan",       35.0116,  135.7681,  40),
    "후쿠오카":   ("Fukuoka",         "Japan",       33.5904,  130.4017,  50),
    "삿포로":     ("Sapporo",         "Japan",       43.0618,  141.3545,  60),
    "오키나와":   ("Okinawa",         "Japan",       26.2124,  127.6809,  80),
    "홍콩":       ("Hong Kong",       "Hong Kong",   22.3193,  114.1694,  50),
    "마카오":     ("Macao",           "Macao",       22.1987,  113.5439,  30),
    "방콕":       ("Bangkok",         "Thailand",    13.7563,  100.5018,  80),
    "다낭":       ("Da Nang",         "Vietnam",     16.0544,  108.2022,  60),
    "하노이":     ("Hanoi",           "Vietnam",     21.0245,  105.8412,  60),
    "호치민":     ("Ho Chi Minh City","Vietnam",     10.8231,  106.6297,  60),
    "싱가포르":   ("Singapore",       "Singapore",    1.3521,  103.8198,  40),
    "타이베이":   ("Taipei",          "Taiwan",      25.0330,  121.5654,  50),
    "타이페이":   ("Taipei",          "Taiwan",      25.0330,  121.5654,  50),
    "파리":       ("Paris",           "France",      48.8566,    2.3522,  60),
    "런던":       ("London",          "UK",          51.5074,   -0.1278,  60),
    "뉴욕":       ("New York",        "USA",         40.7128,  -74.0060,  80),
    "LA":         ("Los Angeles",     "USA",         34.0522, -118.2437,  80),
    "로스앤젤레스":("Los Angeles",    "USA",         34.0522, -118.2437,  80),
    "시드니":     ("Sydney",          "Australia",  -33.8688,  151.2093,  80),
    "멜버른":     ("Melbourne",       "Australia",  -37.8136,  144.9631,  60),
    "바르셀로나": ("Barcelona",       "Spain",       41.3851,    2.1734,  40),
    "로마":       ("Rome",            "Italy",       41.9028,   12.4964,  50),
    "제주":       ("Jeju",        "South Korea",    33.4996,  126.5312,  50),
    "부산":       ("Busan",       "South Korea",    35.1796,  129.0756,  50),
    "경주":       ("Gyeongju",    "South Korea",    35.8562,  129.2246,  40),
    "발리":       ("Bali",            "Indonesia",   -8.3405,  115.0920,  80),
    "쿠알라룸푸르":("Kuala Lumpur",  "Malaysia",     3.1390,  101.6869,  80),
}


# ============================================================
# Structured Output Models
# ============================================================
class RouterIntent(BaseModel):
    intent: Literal["planning", "chitchat"]


class TravelIntent(BaseModel):
    destination: Optional[str] = None
    duration: Optional[str] = None
    preferences: Optional[str] = None


class TripDistricts(BaseModel):
    districts: list[str]


class PlaceForMap(BaseModel):
    name: str          # 표시용 한국어 이름
    english_name: str  # Nominatim 검색용 영어/현지어 이름 (도시명 제외)
    day: int           # 몇 일차 (1부터)


class PlacesPrediction(BaseModel):
    places: list[PlaceForMap]


class ItineraryAnalysis(BaseModel):
    places: list[PlaceForMap]  # 관광지+맛집, 최대 10개
    budget_summary: str         # 예산 한 줄 요약


# ============================================================
# LLM
# ============================================================
def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


def get_structured_llm(model: type[BaseModel]) -> ChatOpenAI:
    return get_llm(temperature=0).with_structured_output(model)


# ============================================================
# Prompts
# ============================================================
def get_router_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system",
         """Classify the user's intent:
- "planning": wants to plan a trip, mentions destination/duration, or is providing travel info
- "chitchat": greeting, thanks, or unrelated conversation"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


def get_analysis_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system",
         """Extract destination and duration from the conversation.
- "X 말고 Y" or "X 대신 Y" means Y overrides X — always use the LATEST value.
- Values can appear in any message in the conversation history.
- Return null if a value is missing or was cancelled."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


def get_skeleton_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system",
         """여행 설계 전문가로서 일차별 동선을 고려해 방문할 핵심 구역을 최대 3개 선정하세요.
(예: 침사추이, 센트럴, 란타우섬)"""),
        ("human", "여행지: {destination} | 기간: {duration} | 선호사항: {preferences}"),
    ])


def get_planner_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system",
         """당신은 전문 여행 플래너입니다. 아래 정보를 바탕으로 상세한 여행 일정을 작성하세요.

[일자별 구역 배정]
{district_schedule}
→ 각 일차는 배정된 구역 내 장소만 추천하세요. 다른 구역 장소를 섞지 마세요.

[실시간 검색 결과 — 관광지]
{attractions}

[실시간 검색 결과 — 맛집/카페]
{food}

[교통/팁]
{tips}

[작성 규칙]
1. 구체적인 장소명 필수 — '쇼핑몰', '야시장' 같은 일반 명사 단독 사용 금지
2. 모든 장소에 `[장소명](https://www.google.com/maps/search/{destination}+장소명): 설명` 형식으로 작성
   - 반드시 링크 직후에 콜론(:)과 설명을 이어서 쓸 것
   - 절대로 링크 뒤에 같은 이름을 평문으로 반복하지 말 것
   - ❌ 잘못된 예: `[신주쿠 교엔](url)신주쿠 교엔: 설명`
   - ✅ 올바른 예: `[신주쿠 교엔](url): 설명`
3. 지리적 위치가 없는 항목(교통 패스, 티켓 등)에는 링크 달지 않기
4. {destination} 외 다른 도시/국가의 장소 추천 금지

포함 항목:
- 📅 일별 관광지 및 활동
- 🍽️ 맛집/카페
- 🏨 숙박 지역
- 🚌 이동 방법
- 💰 예상 비용
- 💡 여행 팁

마크다운 형식으로 작성하세요."""),
        ("human", "여행지: {destination} | 기간: {duration} | 선호사항: {preferences}"),
    ])


def get_itinerary_analysis_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system",
         """여행 일정 텍스트에서 두 가지를 추출하세요.

1. places: 지도에 표시할 장소 (최대 10개)
   - 관광지, 맛집, 카페만 포함
   - 숙소·공항·역·버스·패스·입장권 등 제외
   - name: 한국어 또는 현지 표시명
   - english_name: Nominatim 검색용 영어 이름 (도시명 포함 금지, 장소명만)
     예: "센소지" → "Senso-ji Temple"
         "시부야 스크램블 교차로" → "Shibuya Scramble Crossing"
         "에펠탑" → "Eiffel Tower"
   - day: 몇 일차 (1부터)
   - 중복 제거

2. budget_summary: 예산 핵심만 한 줄
   (예: "1일 약 10만원, 총 3박4일 약 40만원 예상")
   - 예산 정보가 없으면 빈 문자열"""),
        ("human", "여행지: {dest}\n\n일정:\n{itinerary}"),
    ])


def get_places_prediction_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system",
         """여행 검색 결과에서 일정에 포함될 가능성이 높은 관광지와 맛집을 최대 8개 예측하세요.
- name: 한국어 또는 현지 표시명
- english_name: Nominatim 검색용 영어 이름 (장소명만, 도시명 제외)
- day: 구역 순서 기반 예측 일차 (구역 1→1일차, 구역 2→2일차…)
- 중복 없이"""),
        ("human",
         "여행지: {dest}\n구역(1일차부터): {districts}\n\n관광지:\n{attractions}\n\n맛집:\n{food}"),
    ])


# ============================================================
# Google Maps Link Post-Processor
# ============================================================
COUNTRY_MAP = {
    "도쿄": "일본", "오사카": "일본", "교토": "일본", "후쿠오카": "일본",
    "삿포로": "일본", "오키나와": "일본", "파리": "프랑스", "런던": "영국",
    "뉴욕": "미국", "LA": "미국", "로스앤젤레스": "미국", "방콕": "태국",
    "다낭": "베트남", "하노이": "베트남", "호치민": "베트남",
    "싱가포르": "싱가포르", "타이베이": "대만", "타이페이": "대만",
    "홍콩": "홍콩", "마카오": "마카오", "시드니": "호주", "멜버른": "호주",
}


def _country_prefix(destination: str) -> str:
    dest = destination.strip()
    for city, country in COUNTRY_MAP.items():
        if city in dest or dest in city:
            return dest if country == city else f"{country} {dest}"
    return dest


def fix_maps_links(text: str, destination: str) -> str:
    prefix = _country_prefix(destination)
    pattern = r'\[([^\]]+)\]\((https?://(?:www\.)?google\.com/maps/search/([^\s)]+))\)'

    def _fix(match):
        label, raw_query = match.group(1), match.group(3)
        query = unquote(raw_query).replace("+", " ").strip()
        if prefix not in query:
            query = f"{prefix} {query}" if destination not in query else (
                f"{prefix.split()[0]} {query}"
                if len(prefix.split()) > 1 and prefix.split()[0] not in query
                else query
            )
        return f"[{label}](https://www.google.com/maps/search/{quote(query).replace('%20', '+')})"

    return re.sub(pattern, _fix, text)


# ============================================================
# Helpers
# ============================================================
def format_search_results(results: list) -> str:
    if not results:
        return "검색 결과 없음"
    return "\n".join(
        f"• {r.get('title', '')}: {r.get('content', '')[:150]}"
        for r in results if isinstance(r, dict)
    ) or "검색 결과 없음"


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            c.get("text", "") if isinstance(c, dict)
            else c.text if hasattr(c, "text")
            else str(c)
            for c in content
        ]
        return " ".join(p for p in parts if p)
    return content.text if hasattr(content, "text") else str(content)


def to_langchain_messages(history: list) -> list:
    result = []
    for msg in history:
        text = extract_text(msg["content"])
        if msg["role"] == "user":
            result.append(HumanMessage(content=text))
        elif msg["role"] == "assistant":
            result.append(AIMessage(content=text))
    return result


def build_district_schedule(districts: list[str], duration: str) -> str:
    if not districts:
        return "구역 구분 없음"
    return "\n".join(f"- {d}" for d in districts)


# ============================================================
# Geocoding Utilities
# ============================================================
def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def geocode_places(places: list[PlaceForMap], dest: str) -> list[dict]:
    """Nominatim 지오코딩 + 도시 중심 거리 검증."""
    try:
        from geopy.geocoders import Nominatim
    except ImportError:
        return []

    info = DEST_EN_MAP.get(dest)
    dest_en = info[0] if info else None
    dest_country = info[1] if info else None
    center_lat = info[2] if info else None
    center_lng = info[3] if info else None
    max_radius = info[4] if info else 300

    geolocator = Nominatim(user_agent="trip_planner_hf_v3", timeout=5)
    result: list[dict] = []

    for place in places[:10]:
        # 쿼리 우선순위: "장소명, 도시, 국가" → "장소명, 도시" → "장소명"
        queries = []
        if dest_en and dest_country:
            queries.append(f"{place.english_name}, {dest_en}, {dest_country}")
        if dest_en:
            queries.append(f"{place.english_name}, {dest_en}")
        queries.append(place.english_name)

        location = None
        for query in queries:
            try:
                location = geolocator.geocode(query)
                time.sleep(1.1)  # Nominatim rate limit: 1 req/s
                if location and center_lat is not None:
                    dist = _haversine_km(
                        center_lat, center_lng,
                        location.latitude, location.longitude,
                    )
                    if dist > max_radius:
                        location = None
                        continue  # 다른 쿼리로 재시도
                if location:
                    break  # 유효한 좌표 확보
            except Exception:
                time.sleep(1.1)
                continue

        if location:
            result.append({
                "name":     html_module.escape(place.name),
                "name_raw": place.name,
                "en_name":  place.english_name,
                "day":      place.day,
                "lat":      location.latitude,
                "lng":      location.longitude,
            })

    return result


def predict_places_from_search(
    attractions: str,
    food: str,
    dest: str,
    districts: list[str],
) -> list[PlaceForMap]:
    """검색 결과에서 일정에 포함될 장소 미리 예측 (스트리밍 전 병렬 실행용)."""
    try:
        result: PlacesPrediction = (
            get_places_prediction_prompt()
            | get_structured_llm(PlacesPrediction)
        ).invoke({
            "dest": dest,
            "districts": " → ".join(districts),
            "attractions": attractions[:1500],
            "food": food[:800],
        })
        return result.places[:8]
    except Exception as e:
        print(f"[predict_places] {e}")
        return []


def extract_itinerary_analysis(itinerary_text: str, dest: str) -> ItineraryAnalysis:
    """완성된 일정 텍스트에서 장소 목록 + 예산 추출."""
    try:
        return (
            get_itinerary_analysis_prompt()
            | get_structured_llm(ItineraryAnalysis)
        ).invoke({
            "dest": dest,
            "itinerary": itinerary_text[:4000],
        })
    except Exception as e:
        print(f"[analysis] {e}")
        return ItineraryAnalysis(places=[], budget_summary="")


# ============================================================
# Map Building
# ============================================================
def build_folium_map(geocoded: list[dict]) -> str | None:
    if not geocoded:
        return None
    try:
        import folium
    except ImportError:
        return None

    center_lat = sum(p["lat"] for p in geocoded) / len(geocoded)
    center_lng = sum(p["lng"] for p in geocoded) / len(geocoded)

    m = folium.Map(location=[center_lat, center_lng], zoom_start=13)

    # Day-colored markers
    for place in geocoded:
        color = DAY_COLORS[(place["day"] - 1) % len(DAY_COLORS)]
        folium.Marker(
            location=[place["lat"], place["lng"]],
            popup=folium.Popup(
                f"{place['name']} ({place['day']}일차)", max_width=200
            ),
            tooltip=f"{place['day']}일차 | {place['name']}",
            icon=folium.Icon(color=color, icon="map-marker"),
        ).add_to(m)

    # Day-colored polylines (일차별 색상 경로)
    if len(geocoded) > 1:
        by_day: dict[int, list] = defaultdict(list)
        for p in sorted(geocoded, key=lambda x: x["day"]):
            by_day[p["day"]].append([p["lat"], p["lng"]])

        for day_num in sorted(by_day.keys()):
            coords = by_day[day_num]
            if len(coords) > 1:
                color = DAY_COLORS[(day_num - 1) % len(DAY_COLORS)]
                folium.PolyLine(
                    coords, color=color, weight=3, opacity=0.75,
                    tooltip=f"{day_num}일차 경로",
                ).add_to(m)

    return m._repr_html_()


def _budget_html(budget: str) -> str:
    safe = html_module.escape(budget)
    return (
        "<div style='margin-top:8px;padding:10px 14px;"
        "background:rgba(26,86,219,0.12);border:1px solid rgba(26,86,219,0.25);"
        "border-radius:8px;font-size:13px;color:inherit;'>"
        f"💰 <b>예산 요약:</b> {safe}</div>"
    )


def _build_map_html(geocoded: list[dict], total_places: int, budget: str) -> str:
    """folium HTML + 한국어 범례 + 경고 + 예산 요약을 합쳐서 반환."""
    folium_html = build_folium_map(geocoded)
    if not folium_html:
        return MAP_ERROR_HTML

    # 일차별 색상 범례 (한국어)
    COLOR_KO = {
        "red": "#e53e3e", "blue": "#3182ce", "green": "#38a169",
        "purple": "#805ad5", "orange": "#dd6b20", "darkred": "#9b2335",
        "cadetblue": "#4a90a4",
    }
    days_present = sorted({p["day"] for p in geocoded})
    legend_items = "".join(
        f"<span style='margin-right:10px;white-space:nowrap;'>"
        f"<span style='display:inline-block;width:10px;height:10px;"
        f"border-radius:50%;background:{COLOR_KO[DAY_COLORS[(d-1)%len(DAY_COLORS)]]};margin-right:4px;'>"
        f"</span>{d}일차</span>"
        for d in days_present
    )
    legend = (
        f"<div style='padding:6px 10px;font-size:12px;display:flex;flex-wrap:wrap;gap:4px;"
        f"border-bottom:1px solid rgba(128,128,128,0.2);'>"
        f"🗺️ 여행 경로 &nbsp; {legend_items}</div>"
    ) if days_present else ""

    miss = total_places - len(geocoded)
    warning = ""
    if miss > 0:
        warning = (
            f"<p style='font-size:11px;color:#f59e0b;margin:4px 8px 2px;'>"
            f"⚠️ {miss}개 장소의 좌표를 가져오지 못했습니다.</p>"
        )

    budget_part = _budget_html(budget) if budget.strip() else ""
    return (
        legend
        + warning
        + f'<div style="height:360px;overflow:hidden;">{folium_html}</div>'
        + budget_part
    )


# ============================================================
# Pipeline  (병렬화: Stage1‖Stage2, Stage3‖Stage4)
# ============================================================
def prepare_context(user_input: str, history: list) -> tuple:
    """
    Returns:
        quick_response (str | None)
        planner_inputs (dict | None)
        timings (dict)
        dest (str | None)
        districts (list[str])
    """
    timings: dict = {}
    lc_history = to_langchain_messages(history)
    base = {"input": user_input, "chat_history": lc_history}

    # Stage 1‖2 — router + slots 동시 실행
    t1 = time.time()
    stage12 = RunnableParallel(
        route=get_router_prompt() | get_structured_llm(RouterIntent),
        slots=get_analysis_prompt() | get_structured_llm(TravelIntent),
    )
    results12 = stage12.invoke(base)
    route: RouterIntent = results12["route"]
    slots: TravelIntent = results12["slots"]
    timings["stage12_ms"] = round((time.time() - t1) * 1000)
    print(f"[pipeline] router={route.intent!r} dest={slots.destination!r} "
          f"dur={slots.duration!r} ({timings['stage12_ms']}ms)")

    if route.intent == "chitchat":
        return CHITCHAT_RESPONSE, None, timings, None, []

    dest, dur = slots.destination, slots.duration
    if not dest and not dur:
        return MISSING_MESSAGES["both"], None, timings, None, []
    if not dest:
        return MISSING_MESSAGES["destination"], None, timings, None, []
    if not dur:
        return MISSING_MESSAGES["duration"], None, timings, None, []

    # Stage 3‖4 — skeleton + 웹 검색 동시 실행
    t2 = time.time()
    tavily = TavilySearchResults(max_results=3)
    stage34_input = {
        "destination": dest,
        "duration": dur,
        "preferences": slots.preferences or "없음",
    }
    stage34 = RunnableParallel(
        skeleton=RunnableLambda(
            lambda x: (get_skeleton_prompt() | get_structured_llm(TripDistricts)).invoke(x)
        ),
        attractions=RunnableLambda(lambda _: tavily.invoke(f"{dest} 관광지 명소 추천")),
        food=RunnableLambda(lambda _: tavily.invoke(f"{dest} 맛집 카페 레스토랑 추천")),
        tips=RunnableLambda(lambda _: tavily.invoke(f"{dest} 여행 교통 팁 패스")),
    )
    results34 = stage34.invoke(stage34_input)
    skeleton: TripDistricts = results34["skeleton"]
    districts = skeleton.districts[:3]
    timings["stage34_ms"] = round((time.time() - t2) * 1000)
    print(f"[pipeline] districts={districts} ({timings['stage34_ms']}ms)")

    planner_inputs = {
        "destination": dest,
        "duration": dur,
        "preferences": slots.preferences or "없음",
        "district_schedule": build_district_schedule(districts, dur),
        "attractions": format_search_results(results34.get("attractions", [])),
        "food": format_search_results(results34.get("food", [])),
        "tips": format_search_results(results34.get("tips", [])),
    }
    return None, planner_inputs, timings, dest, districts


# ============================================================
# Session Management
# ============================================================
sessions: dict = {}


def _new_session_data() -> dict:
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": 0,
        "timings": [],
    }


def create_session() -> tuple[str, bool]:
    if len(sessions) >= MAX_SESSIONS:
        return "", True
    sid = str(uuid.uuid4())[:8].upper()
    sessions[sid] = _new_session_data()
    return sid, False


def start_new_session() -> tuple[str, str]:
    if len(sessions) >= MAX_SESSIONS:
        return "", "⚠️ 최대 세션 수에 도달했습니다. 세션을 초기화한 후 이용해주세요."
    sid = str(uuid.uuid4())[:8].upper()
    sessions[sid] = _new_session_data()
    return sid, format_session_info(sid)


def reset_sessions() -> tuple[str, list, str]:
    sessions.clear()
    sid, _ = create_session()
    return sid, [], format_session_info(sid)


def format_session_info(current_id: str) -> str:
    if not sessions:
        return "세션 없음"
    lines = []
    for sid, info in sessions.items():
        indicator = "● 현재" if sid == current_id else "○"
        remaining = MAX_MESSAGES_PER_SESSION - info["messages"]
        status = "⚠️ 한도 초과" if remaining <= 0 else f"남은 횟수: {remaining}개"
        lines.append(
            f"{indicator}  [{sid}]\n  생성: {info['created_at']}\n"
            f"  메시지: {info['messages']}개  |  {status}"
        )
    return "\n\n".join(lines)


def record_turn(session_id: str, user_input: str, timings: dict) -> None:
    if session_id not in sessions:
        return
    sessions[session_id]["messages"] += 1
    sessions[session_id]["timings"].append({
        "turn": sessions[session_id]["messages"],
        "user_input": user_input,
        "timestamp": datetime.now().isoformat(),
        **timings,
    })


# ============================================================
# JSON Export
# ============================================================
def export_chat_json(history: list, session_id: str) -> tuple[str, str | None]:
    if not history:
        return "저장할 대화 내역이 없습니다.", None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = {
        "session_id": session_id,
        "saved_at": datetime.now().isoformat(),
        "messages": history,
        "timings": sessions.get(session_id, {}).get("timings", []),
    }
    content = json.dumps(data, ensure_ascii=False, indent=2)

    # 서버 로그용 저장
    try:
        (LOGS_DIR / f"{ts}_{session_id}.json").write_text(content, encoding="utf-8")
    except Exception:
        pass

    # Gradio가 항상 서빙 가능한 temp 파일로 반환 (HF Spaces 호환)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json",
        prefix=f"trip_{session_id}_",
        delete=False, encoding="utf-8",
    )
    tmp.write(content)
    tmp.close()
    return "✅ 저장 완료! 아래 다운로드 버튼을 클릭하세요.", tmp.name


# ============================================================
# Chat Handler  (generator → Gradio streaming)
# Yields 4 outputs: msg_box, chatbot, session_box, map_html
# 지도 지오코딩은 LLM 스트리밍과 병렬로 background thread에서 실행
# ============================================================
def handle_chat(user_input: str, history: list, session_id: str):
    if not user_input or not user_input.strip():
        msg = "가고 싶은 여행 장소와 대략적인 기간을 알려주시면 계획을 짜드리겠습니다. 😊"
        yield "", history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": msg},
        ], format_session_info(session_id), gr.update()
        return

    # 초기 표시
    yield "", history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": "🔍 여행 정보를 수집하고 있습니다..."},
    ], format_session_info(session_id), gr.update()

    timings: dict = {}
    response = ""
    dest = None
    map_out = gr.update()  # no map change by default

    try:
        quick, planner_inputs, timings, dest, districts = prepare_context(
            user_input, history
        )

        if quick is not None:
            response = quick
        else:
            # ── 배경 스레드: 검색 결과 기반 장소 예측 + 지오코딩 시작 ──
            geo_state: dict = {"geocoded": [], "done": False}

            def _bg_geocode() -> None:
                try:
                    predicted = predict_places_from_search(
                        planner_inputs["attractions"],
                        planner_inputs["food"],
                        dest,
                        districts,
                    )
                    geo_state["geocoded"] = geocode_places(predicted, dest)
                except Exception as e:
                    print(f"[bg geocode] {e}")
                finally:
                    geo_state["done"] = True

            geo_thread = threading.Thread(target=_bg_geocode, daemon=True)
            geo_thread.start()

            # 지도 로딩 표시 (LLM 스트리밍 시작과 동시에)
            yield "", history + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": "✍️ 일정을 작성하고 있습니다..."},
            ], format_session_info(session_id), MAP_LOADING_HTML

            # ── LLM 스트리밍 (지오코딩과 병렬) ──
            t_plan = time.time()
            messages = get_planner_prompt().format_messages(**planner_inputs)
            partial = ""
            for chunk in get_llm().stream(messages):
                partial += chunk.content
                yield "", history + [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": partial},
                ], format_session_info(session_id), gr.update()
            timings["planner_ms"] = round((time.time() - t_plan) * 1000)
            response = fix_maps_links(partial, dest)

            # ── 스트리밍 완료 후 지오코딩 최대 5초 대기 ──
            geo_thread.join(timeout=5)

            # ── 완성 일정에서 정확한 장소+예산 추출 ──
            analysis = extract_itinerary_analysis(response, dest)

            # 배경 지오코딩 결과와 병합 (en_name으로 매칭)
            bg_by_en: dict[str, dict] = {
                g["en_name"]: g for g in geo_state["geocoded"]
            }

            final_geocoded: list[dict] = []
            missing_places: list[PlaceForMap] = []

            for place in analysis.places:
                if place.english_name in bg_by_en:
                    g = dict(bg_by_en[place.english_name])
                    g["day"] = place.day  # 정확한 일차 반영
                    final_geocoded.append(g)
                else:
                    missing_places.append(place)

            # 아직 지오코딩 안 된 장소 추가 처리 (최대 5개)
            if missing_places:
                extra = geocode_places(missing_places[:5], dest)
                final_geocoded.extend(extra)

            map_out = _build_map_html(
                final_geocoded,
                len(analysis.places),
                analysis.budget_summary,
            )

    except Exception as e:
        response = f"⚠️ 오류가 발생했습니다: {e}"

    timings["total_ms"] = sum(
        v for k, v in timings.items() if k.endswith("_ms") and k != "total_ms"
    )
    print(f"[pipeline] total={timings['total_ms']}ms | {timings}")

    new_history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response},
    ]

    record_turn(session_id, user_input, timings)

    if sessions.get(session_id, {}).get("messages", 0) >= MAX_MESSAGES_PER_SESSION:
        warning = (
            f"\n\n---\n⚠️ 이 세션의 메시지 한도({MAX_MESSAGES_PER_SESSION}개)에 도달했습니다. "
            "'🆕 새 세션 시작' 버튼을 눌러 새 세션을 시작해주세요."
        )
        new_history[-1] = {
            **new_history[-1],
            "content": new_history[-1]["content"] + warning,
        }

    # 최종 yield — 지도도 함께 업데이트
    yield "", new_history, format_session_info(session_id), map_out


# ============================================================
# Gradio UI
# ============================================================
CSS = """
.gradio-container { max-width: 1200px !important; margin: auto; }
#header { text-align: center; padding: 28px 0 10px; }
#header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; color: #1a56db; }
#header p  { color: #6b7280; font-size: 1rem; margin: 6px 0 0; }
#session-box textarea { font-family: monospace !important; font-size: 12px !important; }

/* 지도 iframe 크기 */
#map-panel iframe {
    width: 100% !important;
    height: 360px !important;
    border: none !important;
}
@media (max-width: 799px) {
    #map-panel iframe { height: 240px !important; }
}

/* 기본(모바일) 세로 1단 */
#main-row {
    display: flex !important;
    flex-direction: column !important;
    gap: 20px !important;
    width: 100% !important;
}
#chat-col, #side-col {
    width: 100% !important;
    flex: 1 1 auto !important;
}

/* PC(≥800px) 좌우 2단 */
@media (min-width: 800px) {
    #main-row {
        flex-direction: row !important;
        flex-wrap: nowrap !important;
    }
    #chat-col {
        flex: 1 1 0% !important;
        min-width: 450px !important;
    }
    #side-col {
        flex: 0 0 300px !important;
        min-width: 300px !important;
    }
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks() as demo:
        current_session = gr.State("")

        gr.HTML("""
        <div id="header">
          <h1>✈️ Travel Planner</h1>
          <p>여행계획을 짜드리는 챗봇입니다</p>
        </div>
        """)

        with gr.Row(equal_height=False, elem_id="main-row"):
            # ── 채팅 + 지도 ─────────────────────────────────────────
            with gr.Column(scale=3, elem_id="chat-col"):

                chatbot = gr.Chatbot(label="대화", height=440)
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="✏️  예) 도쿄 3박 4일 여행 계획 짜줘",
                        show_label=False, scale=5, container=False,
                    )
                    send_btn = gr.Button("전송 ➤", scale=1, variant="primary")
                clear_btn = gr.Button("🗑️ 대화 초기화", variant="primary")

                # 지도: 채팅창 아래에 표시
                map_html = gr.HTML(
                    value=MAP_EMPTY_HTML,
                    elem_id="map-panel",
                )

            # ── 세션 / 저장 패널 ──────────────────────────────────
            with gr.Column(scale=1, min_width=220, elem_id="side-col"):
                gr.Markdown("### 📋 세션 정보")
                gr.HTML(f"""
                <p style="font-size:11px; color:#6b7280; margin:0 0 8px; line-height:1.6;">
                  📌 세션 기준<br>
                  • 세션당 최대 <b>{MAX_MESSAGES_PER_SESSION}개</b> 메시지<br>
                  • 한도 초과 시 새 세션을 시작하세요<br>
                  • 최대 <b>{MAX_SESSIONS}개</b> 세션 동시 유지 가능
                </p>
                """)
                session_box = gr.Textbox(
                    label="현재 세션", lines=6,
                    interactive=False, elem_id="session-box",
                )
                new_session_btn = gr.Button("🆕 새 세션 시작", variant="primary")
                refresh_warning = gr.Markdown(
                    value="⚠️ 최대 세션 수에 도달했습니다.\n세션을 초기화한 후 이용해주세요.",
                    visible=False,
                )
                refresh_btn = gr.Button("🔄 세션 전체 초기화", variant="stop", visible=False)

                gr.HTML("<hr style='margin:16px 0;'>")

                gr.Markdown("### 💾 대화 저장")
                save_btn = gr.Button("📥 JSON으로 저장", variant="primary")
                save_status = gr.Textbox(label="저장 상태", interactive=False, lines=2)
                download_file = gr.File(label="📄 파일 다운로드", visible=False)
                gr.HTML(f"""
                <p style="font-size:12px; color:#9ca3af; margin-top:8px; line-height:1.6;">
                  📂 저장 위치<br>
                  <code style="background:#f3f4f6; padding:2px 6px; border-radius:4px;">
                    {LOGS_DIR.resolve()}
                  </code><br>
                  <span style="margin-top:4px; display:block;">
                    파일명: <code>날짜_세션ID.json</code>
                  </span>
                </p>
                """)

        # ── 이벤트 핸들러 ─────────────────────────────────────────
        def on_load():
            sid, over = create_session()
            visible = gr.update(visible=over)
            return (("", format_session_info(""), visible, visible)
                    if over else
                    (sid, format_session_info(sid),
                     gr.update(visible=False), gr.update(visible=False)))

        demo.load(on_load,
                  outputs=[current_session, session_box, refresh_warning, refresh_btn])

        def on_new_session():
            sid, info = start_new_session()
            if not sid:
                return gr.update(), info, gr.update(visible=True), gr.update(visible=True)
            return sid, info, gr.update(visible=False), gr.update(visible=False)

        new_session_btn.click(on_new_session,
                              outputs=[current_session, session_box,
                                       refresh_warning, refresh_btn])

        def on_refresh():
            sid, hist, info = reset_sessions()
            return sid, hist, info, gr.update(visible=False), gr.update(visible=False)

        refresh_btn.click(on_refresh,
                          outputs=[current_session, chatbot, session_box,
                                   refresh_warning, refresh_btn])

        for trigger in (msg_box.submit, send_btn.click):
            trigger(
                handle_chat,
                inputs=[msg_box, chatbot, current_session],
                outputs=[msg_box, chatbot, session_box, map_html],
            )

        clear_btn.click(
            lambda: ([], MAP_EMPTY_HTML),
            outputs=[chatbot, map_html],
        )

        def on_save(history, session_id):
            status, filepath = export_chat_json(history, session_id)
            return status, gr.update(value=filepath, visible=bool(filepath))

        save_btn.click(on_save, inputs=[chatbot, current_session],
                       outputs=[save_status, download_file])

    return demo


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False, theme=gr.themes.Soft(primary_hue="blue"), css=CSS)
