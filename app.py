import json
import time
import uuid
import re
import html as html_module
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

CHITCHAT_RESPONSE = "이 챗봇은 여행 관련된 응답만 제공 가능합니다. 여행지와 기간을 알려주시면 계획을 짜드릴게요! ✈️"

# Day-color palette for map markers (wraps after 7 days)
DAY_COLORS = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue']

MAP_EMPTY_HTML = (
    "<p style='color:#9ca3af;text-align:center;padding:40px;font-size:14px'>"
    "일정을 생성하면 경로 지도가 표시됩니다 🗺️</p>"
)
MAP_LOADING_HTML = (
    "<p style='text-align:center;padding:40px;font-size:14px'>"
    "🔍 지도 좌표를 가져오는 중입니다... (장소 수에 따라 10~30초 소요)</p>"
)
MAP_ERROR_HTML = (
    "<p style='color:#9ca3af;text-align:center;padding:20px;font-size:13px'>"
    "지도를 표시하지 못했습니다. 각 장소의 Google Maps 링크를 이용해주세요.</p>"
)


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
    name: str          # 표시용 이름 (한국어)
    english_name: str  # Nominatim 검색용 영어 또는 현지어 이름
    day: int           # 몇 일차 (1부터)


class ItineraryAnalysis(BaseModel):
    places: list[PlaceForMap]   # 지도용 장소 (관광지+맛집, 최대 10개)
    budget_summary: str          # 예산 한 줄 요약


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
2. 모든 장소에 `[장소명](https://www.google.com/maps/search/{destination}+장소명)` 형식 링크 포함
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
   - name: 한국어 표시명
   - english_name: Nominatim 지오코딩용 영어 또는 현지어 이름
     (예: "센소지" → "Senso-ji Temple", "에펠탑" → "Eiffel Tower")
   - day: 몇 일차 (1부터)
   - 중복 제거

2. budget_summary: 예산 핵심만 한 줄 (예: "1일 약 10만원, 총 3박4일 약 40만원 예상")
   - 예산 정보가 없으면 빈 문자열 반환"""),
        ("human", "여행지: {dest}\n\n일정:\n{itinerary}"),
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
    """모든 Google Maps 링크에 여행지(+국가) 접두어를 보장한다."""
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
    """Gradio 6 content → plain string."""
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
# Map: Extraction → Geocoding → Rendering
# ============================================================
def extract_itinerary_analysis(itinerary_text: str, dest: str) -> ItineraryAnalysis:
    """일정 텍스트에서 지도용 장소 목록과 예산 요약 추출."""
    prompt = get_itinerary_analysis_prompt()
    llm = get_structured_llm(ItineraryAnalysis)
    try:
        return (prompt | llm).invoke({
            "dest": dest,
            "itinerary": itinerary_text[:4000],
        })
    except Exception as e:
        print(f"[analysis] extraction error: {e}")
        return ItineraryAnalysis(places=[], budget_summary="")


def geocode_places(places: list[PlaceForMap], dest: str) -> list[dict]:
    """Nominatim으로 장소 좌표 변환. 1.1초 간격 준수."""
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
    except ImportError:
        print("[map] geopy not installed")
        return []

    geolocator = Nominatim(user_agent="trip_planner_hf_v2", timeout=5)
    result = []

    for place in places[:10]:  # hard cap: max 10
        try:
            query = place.english_name  # Nominatim은 영어/현지어로만 검색
            location = geolocator.geocode(query)
            if location:
                result.append({
                    "name": html_module.escape(place.name),
                    "day": place.day,
                    "lat": location.latitude,
                    "lng": location.longitude,
                })
            time.sleep(1.1)  # Nominatim rate limit
        except Exception:
            time.sleep(1.1)
            continue

    return result


def build_folium_map(geocoded: list[dict]) -> str | None:
    """folium HTML 지도 생성. 마커 + 경로 폴리라인 포함."""
    if not geocoded:
        return None
    try:
        import folium
    except ImportError:
        print("[map] folium not installed")
        return None

    center_lat = sum(p["lat"] for p in geocoded) / len(geocoded)
    center_lng = sum(p["lng"] for p in geocoded) / len(geocoded)

    m = folium.Map(location=[center_lat, center_lng], zoom_start=13)

    for place in geocoded:
        color = DAY_COLORS[(place["day"] - 1) % len(DAY_COLORS)]
        folium.Marker(
            location=[place["lat"], place["lng"]],
            popup=folium.Popup(
                f"{place['name']} ({place['day']}일차)", max_width=200
            ),
            tooltip=place["name"],
            icon=folium.Icon(color=color, icon="info-sign"),
        ).add_to(m)

    if len(geocoded) > 1:
        coords = [[p["lat"], p["lng"]] for p in geocoded]
        folium.PolyLine(coords, color="#6b7280", weight=2.5, opacity=0.8).add_to(m)

    return m._repr_html_()


def render_map(itinerary_data: dict) -> tuple:
    """
    .then()으로 호출되는 메인 지오코딩+렌더링 함수.
    Returns: (map_html_str, accordion_gr_update)
    """
    if not itinerary_data or not itinerary_data.get("text") or not itinerary_data.get("dest"):
        return gr.update(), gr.update(open=False)

    text = itinerary_data["text"]
    dest = itinerary_data["dest"]

    try:
        analysis = extract_itinerary_analysis(text, dest)
        places = analysis.places
        budget = analysis.budget_summary.strip()

        if not places:
            out = MAP_ERROR_HTML
            if budget:
                out += _budget_html(budget)
            return out, gr.update(open=True)

        geocoded = geocode_places(places, dest)

        map_html = build_folium_map(geocoded)

        if not map_html:
            out = MAP_ERROR_HTML
        else:
            partial_miss = len(places) - len(geocoded)
            warning = ""
            if partial_miss > 0:
                warning = (
                    f"<p style='font-size:11px;color:#f59e0b;margin:4px 8px 0;'>"
                    f"⚠️ {partial_miss}개 장소의 좌표를 가져오지 못했습니다.</p>"
                )
            out = warning + f'<div style="height:400px;overflow:hidden;">{map_html}</div>'

        if budget:
            out += _budget_html(budget)

        return out, gr.update(open=True)

    except Exception as e:
        print(f"[map] render error: {e}")
        return MAP_ERROR_HTML, gr.update(open=True)


def _budget_html(budget: str) -> str:
    safe = html_module.escape(budget)
    return (
        f"<div style='margin-top:8px;padding:10px 14px;"
        f"background:#f0fdf4;border:1px solid #bbf7d0;"
        f"border-radius:8px;font-size:13px;color:#166534;'>"
        f"💰 <b>예산 요약:</b> {safe}</div>"
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
    """
    timings: dict = {}
    t0 = time.time()
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
        return CHITCHAT_RESPONSE, None, timings, None

    dest, dur = slots.destination, slots.duration
    if not dest and not dur:
        return MISSING_MESSAGES["both"], None, timings, None
    if not dest:
        return MISSING_MESSAGES["destination"], None, timings, None
    if not dur:
        return MISSING_MESSAGES["duration"], None, timings, None

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
    return None, planner_inputs, timings, dest


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
    filepath = LOGS_DIR / f"{ts}_{session_id}.json"
    data = {
        "session_id": session_id,
        "saved_at": datetime.now().isoformat(),
        "messages": history,
        "timings": sessions.get(session_id, {}).get("timings", []),
    }
    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"✅ 저장 완료: {filepath}", str(filepath)


# ============================================================
# Chat Handler  (generator → Gradio streaming)
# Yields 4 outputs: msg_box, chatbot, session_box, last_itinerary_state
# ============================================================
def handle_chat(user_input: str, history: list, session_id: str):
    empty_state: dict = {}

    if not user_input or not user_input.strip():
        msg = "가고 싶은 여행 장소와 대략적인 기간을 알려주시면 계획을 짜드리겠습니다. 😊"
        yield "", history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": msg},
        ], format_session_info(session_id), empty_state
        return

    # 사용자 메시지 즉시 표시
    yield "", history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": "🔍 여행 정보를 수집하고 있습니다..."},
    ], format_session_info(session_id), empty_state

    timings: dict = {}
    response = ""
    dest = None
    try:
        quick_response, planner_inputs, timings, dest = prepare_context(user_input, history)

        if quick_response is not None:
            response = quick_response
        else:
            t_plan = time.time()
            messages = get_planner_prompt().format_messages(**planner_inputs)
            partial = ""
            for chunk in get_llm().stream(messages):
                partial += chunk.content
                yield "", history + [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": partial},
                ], format_session_info(session_id), empty_state
            timings["planner_ms"] = round((time.time() - t_plan) * 1000)
            response = fix_maps_links(partial, dest)

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

    # 최종 yield: 지도용 state 설정 (planning 응답일 때만)
    final_state = {"text": response, "dest": dest} if dest else empty_state
    yield "", new_history, format_session_info(session_id), final_state


# ============================================================
# Gradio UI
# ============================================================
CSS = """
.gradio-container { max-width: 1200px !important; margin: auto; }
#header { text-align: center; padding: 28px 0 10px; }
#header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; color: #1a56db; }
#header p  { color: #6b7280; font-size: 1rem; margin: 6px 0 0; }
#session-box textarea { font-family: monospace !important; font-size: 12px !important; }

/* [기본 상태: 모바일 및 좁은 화면] 세로 1열로 나열 */
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

/* [PC 및 태블릿 화면 - 너비 800px 이상] 좌우 2단 배치 적용 */
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

/* 지도 iframe 크기 제어 */
#map-accordion iframe {
    width: 100% !important;
    height: 400px !important;
    border: none !important;
}
@media (max-width: 799px) {
    #map-accordion iframe {
        height: 260px !important;
    }
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks() as demo:
        current_session = gr.State("")
        last_itinerary = gr.State({})   # {"text": ..., "dest": ...} | {}

        gr.HTML("""
        <div id="header">
          <h1>✈️ Travel Planner</h1>
          <p>여행계획을 짜드리는 챗봇입니다</p>
        </div>
        """)

        with gr.Row(equal_height=False, elem_id="main-row"):
            # ── 채팅 영역 ─────────────────────────────────────────
            with gr.Column(scale=3, elem_id="chat-col"):
                chatbot = gr.Chatbot(label="대화", height=520)
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="✏️  예) 도쿄 3박 4일 여행 계획 짜줘",
                        show_label=False, scale=5, container=False,
                    )
                    send_btn = gr.Button("전송 ➤", scale=1, variant="primary")
                clear_btn = gr.Button("🗑️ 대화 초기화", variant="primary")

                # 지도 아코디언 (일정 생성 후 자동 오픈)
                with gr.Accordion(
                    "🗺️ 여행 경로 지도", open=False, elem_id="map-accordion"
                ) as map_accordion:
                    map_html = gr.HTML(value=MAP_EMPTY_HTML)

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
                    (sid, format_session_info(sid), gr.update(visible=False), gr.update(visible=False)))

        demo.load(on_load, outputs=[current_session, session_box, refresh_warning, refresh_btn])

        def on_new_session():
            sid, info = start_new_session()
            if not sid:
                return gr.update(), info, gr.update(visible=True), gr.update(visible=True)
            return sid, info, gr.update(visible=False), gr.update(visible=False)

        new_session_btn.click(on_new_session,
                              outputs=[current_session, session_box, refresh_warning, refresh_btn])

        def on_refresh():
            sid, hist, info = reset_sessions()
            return sid, hist, info, gr.update(visible=False), gr.update(visible=False)

        refresh_btn.click(on_refresh,
                          outputs=[current_session, chatbot, session_box, refresh_warning, refresh_btn])

        def show_map_loading(itinerary_data: dict):
            """Streaming 완료 직후 지도 로딩 상태 표시."""
            if not itinerary_data or not itinerary_data.get("text"):
                return gr.update(), gr.update()
            return MAP_LOADING_HTML, gr.update(open=True)

        for trigger in (msg_box.submit, send_btn.click):
            chat_event = trigger(
                handle_chat,
                inputs=[msg_box, chatbot, current_session],
                outputs=[msg_box, chatbot, session_box, last_itinerary],
            )
            # 스트리밍 완료 후 즉시 로딩 표시
            chat_event.then(
                show_map_loading,
                inputs=[last_itinerary],
                outputs=[map_html, map_accordion],
            # 그 다음 실제 지오코딩 + 렌더링
            ).then(
                render_map,
                inputs=[last_itinerary],
                outputs=[map_html, map_accordion],
            )

        def on_clear():
            return [], MAP_EMPTY_HTML, gr.update(open=False), {}

        clear_btn.click(
            on_clear,
            outputs=[chatbot, map_html, map_accordion, last_itinerary],
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
