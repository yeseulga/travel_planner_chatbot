import json
import time
import uuid
import re
from urllib.parse import quote, unquote
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
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
    """구역 리스트 → '1~2일차: 침사추이, 3일차: 센트럴, ...' 형식 문자열."""
    if not districts:
        return "구역 구분 없음"
    lines = [f"- {d}" for d in districts]
    return "\n".join(lines)


# ============================================================
# Pipeline
# ============================================================
def prepare_context(user_input: str, history: list) -> tuple:
    """
    LLM 3단계(router → slot → skeleton) + 병렬 웹 검색 수행.

    Returns:
        quick_response (str | None)  — chitchat/missing info면 즉시 반환할 메시지
        planner_inputs (dict | None) — 플래너에 넘길 입력값
        timings (dict)               — 각 단계 소요시간(ms)
        dest (str | None)            — 여행지 (링크 후처리에 사용)
    """
    timings: dict = {}
    t0 = time.time()
    lc_history = to_langchain_messages(history)
    base = {"input": user_input, "chat_history": lc_history}

    # Stage 1 — 의도 분류
    route: RouterIntent = (get_router_prompt() | get_structured_llm(RouterIntent)).invoke(base)
    timings["router_ms"] = round((time.time() - t0) * 1000)
    print(f"[pipeline] router={route.intent!r} ({timings['router_ms']}ms)")

    if route.intent == "chitchat":
        return CHITCHAT_RESPONSE, None, timings, None

    # Stage 2 — 슬롯 추출
    t1 = time.time()
    slots: TravelIntent = (get_analysis_prompt() | get_structured_llm(TravelIntent)).invoke(base)
    timings["slot_ms"] = round((time.time() - t1) * 1000)
    print(f"[pipeline] dest={slots.destination!r} dur={slots.duration!r} ({timings['slot_ms']}ms)")

    dest, dur = slots.destination, slots.duration
    if not dest and not dur:
        return MISSING_MESSAGES["both"], None, timings, None
    if not dest:
        return MISSING_MESSAGES["destination"], None, timings, None
    if not dur:
        return MISSING_MESSAGES["duration"], None, timings, None

    # Stage 3 — 구역 선정
    t2 = time.time()
    skeleton: TripDistricts = (get_skeleton_prompt() | get_structured_llm(TripDistricts)).invoke({
        "destination": dest,
        "duration": dur,
        "preferences": slots.preferences or "없음",
    })
    districts = skeleton.districts[:3]
    timings["skeleton_ms"] = round((time.time() - t2) * 1000)
    print(f"[pipeline] districts={districts} ({timings['skeleton_ms']}ms)")

    # Stage 4 — 병렬 웹 검색 (구역별이 아닌 목적별 3개 쿼리)
    t3 = time.time()
    tavily = TavilySearchResults(max_results=3)

    def search(query: str) -> RunnableLambda:
        return RunnableLambda(lambda _, q=query: tavily.invoke(q))

    results = RunnableParallel(
        attractions=search(f"{dest} 관광지 명소 추천"),
        food=search(f"{dest} 맛집 카페 레스토랑 추천"),
        tips=search(f"{dest} 여행 교통 팁 패스"),
    ).invoke({})
    timings["search_ms"] = round((time.time() - t3) * 1000)
    print(f"[pipeline] search done ({timings['search_ms']}ms)")

    planner_inputs = {
        "destination": dest,
        "duration": dur,
        "preferences": slots.preferences or "없음",
        # 구역은 검색 대신 플래너 지시문으로만 전달
        "district_schedule": build_district_schedule(districts, dur),
        "attractions": format_search_results(results.get("attractions", [])),
        "food": format_search_results(results.get("food", [])),
        "tips": format_search_results(results.get("tips", [])),
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
        lines.append(f"{indicator}  [{sid}]\n  생성: {info['created_at']}\n  메시지: {info['messages']}개  |  {status}")
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
# ============================================================
def handle_chat(user_input: str, history: list, session_id: str):
    if not user_input or not user_input.strip():
        msg = "가고 싶은 여행 장소와 대략적인 기간을 알려주시면 계획을 짜드리겠습니다. 😊"
        new_history = history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": msg},
        ]
        yield "", new_history, format_session_info(session_id)
        return

    # 사용자 메시지 즉시 표시
    yield "", history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": "🔍 여행 정보를 수집하고 있습니다..."},
    ], format_session_info(session_id)

    timings: dict = {}
    response = ""
    try:
        quick_response, planner_inputs, timings, dest = prepare_context(user_input, history)

        if quick_response is not None:
            response = quick_response
        else:
            # 플래너 스트리밍
            t_plan = time.time()
            messages = get_planner_prompt().format_messages(**planner_inputs)
            partial = ""
            for chunk in get_llm().stream(messages):
                partial += chunk.content
                yield "", history + [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": partial},
                ], format_session_info(session_id)
            timings["planner_ms"] = round((time.time() - t_plan) * 1000)
            response = fix_maps_links(partial, dest)

    except Exception as e:
        response = f"⚠️ 오류가 발생했습니다: {e}"

    timings["total_ms"] = sum(v for k, v in timings.items() if k.endswith("_ms") and k != "total_ms")
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
        new_history[-1] = {**new_history[-1], "content": new_history[-1]["content"] + warning}

    yield "", new_history, format_session_info(session_id)


# ============================================================
# Gradio UI
# ============================================================
CSS = """
.gradio-container { max-width: 1200px !important; margin: auto; }
#header { text-align: center; padding: 28px 0 10px; }
#header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; color: #1a56db; }
#header p  { color: #6b7280; font-size: 1rem; margin: 6px 0 0; }
#session-box textarea { font-family: monospace !important; font-size: 12px !important; }
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

        with gr.Row(equal_height=False):
            # ── 채팅 영역 ─────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="대화", height=520)
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="✏️  예) 도쿄 3박 4일 여행 계획 짜줘",
                        show_label=False, scale=5, container=False,
                    )
                    send_btn = gr.Button("전송 ➤", scale=1, variant="primary")
                clear_btn = gr.Button("🗑️ 대화 초기화", variant="primary")

            # ── 세션 / 저장 패널 ──────────────────────────────────
            with gr.Column(scale=1, min_width=220):
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

        for trigger in (msg_box.submit, send_btn.click):
            trigger(handle_chat,
                    inputs=[msg_box, chatbot, current_session],
                    outputs=[msg_box, chatbot, session_box])

        clear_btn.click(lambda: [], outputs=[chatbot])

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
