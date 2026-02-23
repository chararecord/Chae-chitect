"""
Chae-chitect — FastAPI + SSE 백엔드 (Human-in-the-Loop Review Chat)
"""

import asyncio
import json
import logging
import re
import sqlite3
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("chae-chitect")

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

BASE_DIR = _BACKEND_DIR.parent

# ── 세션 스토어 ──────────────────────────────────────────────
_SESSIONS_DB = BASE_DIR / "backend" / "data" / "sessions.db"
_db_lock     = threading.Lock()


def _init_db() -> None:
    _SESSIONS_DB.parent.mkdir(parents=True, exist_ok=True)
    with _db_lock:
        conn = sqlite3.connect(str(_SESSIONS_DB))
        try:
            conn.executescript("""
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS sessions (
                    thread_id     TEXT PRIMARY KEY,
                    title         TEXT NOT NULL,
                    created_at    TEXT NOT NULL,
                    completed_at  TEXT,
                    status        TEXT NOT NULL DEFAULT 'running',
                    features_json TEXT DEFAULT '[]',
                    tech_json     TEXT DEFAULT '[]',
                    obsidian_text TEXT DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id  TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    msg_type   TEXT DEFAULT 'normal',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (thread_id) REFERENCES sessions(thread_id) ON DELETE CASCADE
                );
            """)
            conn.commit()
        finally:
            conn.close()


def _db(sql: str, params: tuple = (), *, fetch: str = "none"):
    with _db_lock:
        conn = sqlite3.connect(str(_SESSIONS_DB))
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            cur = conn.execute(sql, params)
            conn.commit()
            if fetch == "one":
                row = cur.fetchone()
                return dict(row) if row else None
            if fetch == "all":
                return [dict(r) for r in cur.fetchall()]
            return cur.rowcount
        finally:
            conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_create(thread_id: str, title: str) -> None:
    _db(
        "INSERT OR IGNORE INTO sessions (thread_id, title, created_at) VALUES (?, ?, ?)",
        (thread_id, title[:80], _now()),
    )


def _session_msg(thread_id: str, role: str, content: str, msg_type: str = "normal") -> None:
    _db(
        "INSERT INTO messages (thread_id, role, content, msg_type, created_at) VALUES (?, ?, ?, ?, ?)",
        (thread_id, role, content, msg_type, _now()),
    )


def _session_save_node(thread_id: str, node_name: str, result: dict) -> None:
    """node_done 이벤트 발생 시 각 노드 결과를 즉시 DB에 저장한다."""
    if node_name == "decomposer":
        features = result.get("features", [])
        _db("UPDATE sessions SET features_json=? WHERE thread_id=?",
            (json.dumps(features, ensure_ascii=False), thread_id))
    elif node_name == "tech_matcher":
        tech = result.get("tech", [])
        _db("UPDATE sessions SET tech_json=? WHERE thread_id=?",
            (json.dumps(tech, ensure_ascii=False), thread_id))
    elif node_name == "obsidian_formatter":
        obsidian = result.get("obsidian", "")
        _db("UPDATE sessions SET obsidian_text=? WHERE thread_id=?",
            (obsidian, thread_id))


def _session_complete(thread_id: str, features: list, tech: list, obsidian: str) -> None:
    _db(
        "UPDATE sessions SET status=?, completed_at=?, features_json=?, tech_json=?, obsidian_text=? "
        "WHERE thread_id=?",
        (
            "completed", _now(),
            json.dumps(features, ensure_ascii=False),
            json.dumps(tech, ensure_ascii=False),
            obsidian, thread_id,
        ),
    )


_init_db()
# ────────────────────────────────────────────────────────────

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app import build_architect_graph

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

architect_app = build_architect_graph()

NODES = ["decomposer", "tech_matcher", "obsidian_formatter"]
NODE_KO = {
    "decomposer":         "아이디어 분해",
    "tech_matcher":       "기술 스택 매칭",
    "obsidian_formatter": "Obsidian 문서 생성",
}

# ── 인메모리 상태 ─────────────────────────────
_continue_events:  dict[str, asyncio.Event] = {}  # thread_id → 재개 신호
_review_history:   dict[str, list]          = {}  # thread_id → 대화 이력
_review_node:      dict[str, str]           = {}  # thread_id → 현재 리뷰 중인 노드


# ── 유틸 ──────────────────────────────────────

def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def build_result(node_name: str, node_data: dict) -> dict:
    retry_key = {"decomposer": "decomposer_retries", "tech_matcher": "tech_retries"}.get(node_name)
    retries = node_data.get(retry_key, 0) if retry_key else 0

    if node_name == "decomposer":
        return {"features": node_data.get("feature_breakdown", []), "retries": retries}
    if node_name == "tech_matcher":
        return {"tech": node_data.get("tech_requirements", []), "retries": retries}
    if node_name == "obsidian_formatter":
        return {"obsidian": node_data.get("obsidian_template", "")}
    return {}


PROCEED_RULES = """
다음 단계 진행 감지 규칙 (반드시 대화 맥락 전체를 보고 판단하세요):

[1단계 — 진행 의사 확인: <ask_proceed/>]
사용자가 현재 검토를 끝내고 다음 단계로 넘어가겠다는 의도를 명시적으로 드러낼 때만 사용합니다.
  해당 예시: "다음으로 넘어가자", "이제 충분해", "기술 스택으로 가자", "진행해줘", "이걸로 확정할게"
  비해당 예시: "응", "그래", "맞아", "ㅇㅇ", "오케이" — 이것들은 대화 흐름 중 맞장구일 뿐입니다.
→ 해당하는 경우: "다음 단계로 넘어갈까요?" 라고 자연스럽게 물어보고 응답 끝에 <ask_proceed/> 추가

[2단계 — 자동 진행: <auto_proceed/>]
아래 두 조건을 모두 반드시 충족할 때만 사용합니다:
  조건 A: 대화 기록에서 바로 직전 AI 응답이 "다음 단계로 넘어갈까요?" 또는 그와 동일한 의미의 진행 확인 질문이었어야 합니다.
  조건 B: 그 질문에 대해 사용자가 "응", "그래", "ㅇㅇ", "예", "go" 등 명확하게 긍정한 경우입니다.
→ 두 조건 모두 충족 시만: 긍정적으로 마무리하고 응답 끝에 <auto_proceed/> 추가

주의:
- 두 태그를 동시에 쓰지 마세요. 응답 끝에 한 번만 추가하세요.
- 검토 대화 도중 나오는 단순 맞장구, 피처 관련 질문·답변, 수정 확인 등은 절대 진행 신호로 해석하지 마세요.
- 확신이 없으면 <ask_proceed/>도 <auto_proceed/>도 쓰지 말고 대화를 이어가세요.
"""

def _build_opening_message(node: str, state_values: dict) -> str:
    """리뷰 시작 시 AI가 먼저 건네는 첫 마디 (재시도 여부에 따라 다른 메시지)"""
    if node == "decomposer":
        features = state_values.get("feature_breakdown", [])
        retries  = state_values.get("decomposer_retries", 1)
        count = len(features)
        names = ", ".join(f['name'] for f in features[:3]) if features else ""
        tail  = " 등" if count > 3 else ""

        if retries > 1:
            return (
                f"품질 검증 후 **{retries - 1}회 재시도**한 결과, **{count}개**의 피처가 도출됐습니다 "
                f"({names}{tail}).\n\n"
                "결과를 확인하고 추가로 수정하고 싶은 부분이 있으시면 말씀해 주세요 😊"
            )
        return (
            f"아이디어 분해가 완료됐어요! 총 **{count}개**의 피처가 도출됐습니다 "
            f"({names}{tail}).\n\n"
            "왼쪽 패널에서 결과를 확인해보세요. "
            "빠진 기능이 있거나 수정하고 싶은 피처가 있으면 편하게 말씀해 주세요 😊"
        )

    if node == "tech_matcher":
        tech    = state_values.get("tech_requirements", [])
        retries = state_values.get("tech_retries", 1)
        count = len(tech)

        if retries > 1:
            return (
                f"품질 검증 후 **{retries - 1}회 재시도**한 결과, **{count}개** 기술 추천이 완성됐습니다.\n\n"
                "결과를 확인하고 변경이 필요한 기술이 있으시면 말씀해 주세요 🛠️"
            )
        return (
            f"기술 스택 매칭이 완료됐어요! 총 **{count}개** 항목의 기술 추천이 나왔습니다.\n\n"
            "왼쪽 패널에서 확인해보세요. "
            "특정 기술을 다른 것으로 바꾸고 싶거나 이유가 궁금한 게 있으시면 물어봐 주세요 🛠️"
        )

    return "결과를 확인해보세요. 수정이나 질문이 있으시면 말씀해 주세요."


def build_review_system(node: str, state_values: dict) -> str:
    """노드별 리뷰 대화용 시스템 프롬프트"""
    if node == "decomposer":
        features_json = json.dumps(state_values.get("feature_breakdown", []), ensure_ascii=False, indent=2)
        return f"""당신은 Plugin Architect입니다. 아이디어 분해 결과를 사용자와 함께 검토하고 있습니다.

현재 Feature 목록:
{features_json}

역할:
- 사용자의 질문에 답하거나 피처를 설명하세요.
- 수정/추가/삭제 요청 시 반영하고, 반드시 변경 사항을 아래 태그로 반환하세요:

<updated_features>
[{{"id": "F1", "name": "피처명", "description": "설명"}}]
</updated_features>

변경이 없으면 태그 없이 대화만 하세요. 항상 한국어로 답하세요.

{PROCEED_RULES}"""

    if node == "tech_matcher":
        features_json = json.dumps(state_values.get("feature_breakdown", []), ensure_ascii=False, indent=2)
        tech_json     = json.dumps(state_values.get("tech_requirements", []), ensure_ascii=False, indent=2)
        return f"""당신은 Tech Stack Advisor입니다. 기술 스택 추천 결과를 사용자와 함께 검토하고 있습니다.

Feature 목록:
{features_json}

현재 기술 스택 추천:
{tech_json}

역할:
- 각 기술 선택 이유를 설명하거나 대안을 제안하세요.
- 변경 요청 시 반영하고, 반드시 아래 태그로 반환하세요:

<updated_tech>
["F1에 대한 기술 추천", "F2에 대한 기술 추천"]
</updated_tech>

변경이 없으면 태그 없이 대화만 하세요. 항상 한국어로 답하세요.

{PROCEED_RULES}"""

    return "결과를 사용자와 함께 검토하세요. 한국어로 답하세요."


def parse_updated_features(response: str) -> list | None:
    m = re.search(r"<updated_features>(.*?)</updated_features>", response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            return None
    return None


def parse_updated_tech(response: str) -> list | None:
    m = re.search(r"<updated_tech>(.*?)</updated_tech>", response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            return None
    return None


# ── async 스트림 래퍼 ─────────────────────────

async def iter_graph_async(inputs, config):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _worker():
        try:
            for chunk in architect_app.stream(
                inputs, config=config, stream_mode=["updates", "messages"]
            ):
                asyncio.run_coroutine_threadsafe(queue.put(("chunk", chunk)), loop).result()
        except Exception as e:
            asyncio.run_coroutine_threadsafe(queue.put(("error", e)), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop).result()

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        kind, data = await queue.get()
        if kind == "done":
            return
        if kind == "error":
            raise data
        yield data


# ── 파이프라인 스트리밍 ────────────────────────

def _quality_fail_reason(node: str, state_values: dict) -> str:
    """현재 그래프 state를 보고 품질 실패 이유를 한국어로 설명"""
    if node == "decomposer":
        features = state_values.get("feature_breakdown", [])
        if len(features) < 3:
            return f"피처 수 부족 ({len(features)}개 — 최소 3개 필요)"
        missing = [f.get("id", "?") for f in features
                   if not (f.get("id") and f.get("name") and f.get("description"))]
        if missing:
            return f"필수 필드(id/name/description) 누락 — 피처 {', '.join(missing)}"
        return f"피처 {len(features)}개 — 예상치 못한 품질 실패"
    if node == "tech_matcher":
        features = state_values.get("feature_breakdown", [])
        tech     = state_values.get("tech_requirements", [])
        if len(tech) < len(features):
            return f"기술 추천 부족 ({len(tech)}개 생성 — 피처 {len(features)}개 필요)"
        return f"기술 스택 {len(tech)}개 — 예상치 못한 품질 실패"
    return f"'{node}' 품질 기준 미달"


async def stream_pipeline(message: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    yield sse({"type": "start", "nodes": NODES, "node_ko": NODE_KO, "thread_id": thread_id})

    inputs = {
        "messages": [HumanMessage(content=message)],
        "feature_breakdown": [],
        "tech_requirements": [],
        "obsidian_template": "",
        "decomposer_retries": 0,
        "tech_retries": 0,
    }

    seen_nodes: set[str] = set()   # 이미 실행된 노드 추적 (재시도 감지용)
    final_state = None

    while True:
        current_node  = None   # ← 루프마다 리셋해야 재시도 node_start가 정상 발송됨
        last_done_node = None

        async for chunk in iter_graph_async(inputs, config):
            mode, data = chunk

            if mode == "messages":
                token_msg, metadata = data
                node = metadata.get("langgraph_node", "")
                if node not in NODES:
                    continue
                if node != current_node:
                    current_node = node
                    is_retry = node in seen_nodes
                    if is_retry:
                        # 체크포인트의 현재 state로 품질 실패 이유 계산
                        try:
                            cur_state = architect_app.get_state(config)
                            reason = _quality_fail_reason(node, cur_state.values)
                        except Exception:
                            reason = "품질 기준 미달"
                        yield sse({"type": "node_start", "node": node,
                                   "retry": True, "reason": reason})
                    else:
                        yield sse({"type": "node_start", "node": node, "retry": False})
                    seen_nodes.add(node)
                if isinstance(token_msg, AIMessageChunk) and token_msg.content:
                    yield sse({"type": "token", "node": node, "content": token_msg.content})

            elif mode == "updates":
                node_name = next(iter(data))
                if node_name not in NODES:
                    continue
                result = build_result(node_name, data[node_name])
                yield sse({"type": "node_done", "node": node_name, "result": result})
                _session_save_node(thread_id, node_name, result)
                last_done_node = node_name

        graph_state = architect_app.get_state(config)

        if graph_state.next:
            # 리뷰 대화 초기화
            _review_node[thread_id]    = last_done_node
            _review_history[thread_id] = []

            evt = asyncio.Event()
            _continue_events[thread_id] = evt

            # 리뷰 시작 이벤트 — 초기 안내 메시지 포함
            node_ko   = NODE_KO.get(last_done_node, last_done_node)
            next_node = list(graph_state.next)[0]
            next_ko   = NODE_KO.get(next_node, "")
            state_vals = graph_state.values

            opening = _build_opening_message(last_done_node, state_vals)

            yield sse({
                "type":    "review_start",
                "node":    last_done_node,
                "next":    next_node,
                "guide":   f"**{node_ko}** 결과를 검토해보세요. 수정하고 싶은 부분이 있으면 말씀해 주세요. 만족하시면 **'{next_ko} 시작'** 버튼을 눌러주세요.",
                "opening": opening,
            })
            if opening:
                _session_msg(thread_id, "assistant", opening, "review_opening")

            await evt.wait()
            _continue_events.pop(thread_id, None)
            _review_history.pop(thread_id, None)
            _review_node.pop(thread_id, None)

            yield sse({"type": "review_end"})
            inputs = None
        else:
            final_state = graph_state
            break

    if final_state:
        vals = final_state.values
        _session_complete(
            thread_id,
            features=vals.get("feature_breakdown", []),
            tech=vals.get("tech_requirements", []),
            obsidian=vals.get("obsidian_template", ""),
        )

    yield sse({"type": "done"})


# ── 리뷰 대화 ────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class ReviewRequest(BaseModel):
    message: str


async def stream_review_response(thread_id: str, user_message: str):
    import logging


    try:
        config      = {"configurable": {"thread_id": thread_id}}
        node        = _review_node.get(thread_id, "")
        graph_state = architect_app.get_state(config)
        history     = _review_history.setdefault(thread_id, [])

        logger.info(f"[review] thread={thread_id[:8]} node={node!r} history_len={len(history)}")

        system_prompt = build_review_system(node, graph_state.values)

        # 메시지 구성
        lc_messages = [SystemMessage(content=system_prompt)]
        for turn in history:
            if turn["role"] == "user":
                lc_messages.append(HumanMessage(content=turn["content"]))
            else:
                lc_messages.append(AIMessage(content=turn["content"]))
        lc_messages.append(HumanMessage(content=user_message))

        history.append({"role": "user", "content": user_message})
        _session_msg(thread_id, "user", user_message, "review")

        llm = ChatOpenAI(model="gpt-4o", temperature=0.3, streaming=True)
        full_response = ""

        async for chunk in llm.astream(lc_messages):
            if chunk.content:
                full_response += chunk.content
                yield sse({"type": "token", "content": chunk.content})

        logger.info(f"[review] LLM responded {len(full_response)} chars")
        history.append({"role": "assistant", "content": full_response})
        _session_msg(thread_id, "assistant", full_response, "review")

        # 상태 업데이트 파싱 — 변경 즉시 DB에도 반영
        if node == "decomposer":
            updated = parse_updated_features(full_response)
            if updated:
                architect_app.update_state(config, {"feature_breakdown": updated})
                yield sse({"type": "state_update", "node": node, "result": {"features": updated}})
                _session_save_node(thread_id, "decomposer", {"features": updated})

        elif node == "tech_matcher":
            updated = parse_updated_tech(full_response)
            if updated:
                architect_app.update_state(config, {"tech_requirements": updated})
                yield sse({"type": "state_update", "node": node, "result": {"tech": updated}})
                _session_save_node(thread_id, "tech_matcher", {"tech": updated})

        # 진행 의도 감지
        if "<auto_proceed/>" in full_response:
            yield sse({"type": "auto_proceed"})
        elif "<ask_proceed/>" in full_response:
            yield sse({"type": "ask_proceed"})

    except Exception as e:
        logger.error(f"[review] 오류 발생: {e}", exc_info=True)
        yield sse({"type": "error", "message": str(e)})

    finally:
        yield sse({"type": "done"})


# ── 엔드포인트 ────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    thread_id = str(uuid.uuid4())
    _session_create(thread_id, req.message)
    _session_msg(thread_id, "user", req.message)
    return StreamingResponse(
        stream_pipeline(req.message, thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/review/{thread_id}")
async def review(thread_id: str, req: ReviewRequest):
    return StreamingResponse(
        stream_review_response(thread_id, req.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/continue/{thread_id}")
async def continue_pipeline(thread_id: str):
    evt = _continue_events.get(thread_id)
    if evt:
        evt.set()
        return {"ok": True}
    return {"ok": False, "error": "대기 중인 interrupt 없음"}


# ── 기존 세션 이어가기 ────────────────────────────────────────

async def stream_session_chat(thread_id: str, message: str):
    """완료된 세션에서 기존 결과를 컨텍스트로 LLM과 대화를 이어간다."""
    s = _db(
        "SELECT features_json, tech_json, obsidian_text FROM sessions WHERE thread_id = ?",
        (thread_id,), fetch="one",
    )
    if not s:
        yield sse({"type": "error", "message": "세션을 찾을 수 없습니다"})
        yield sse({"type": "done"})
        return

    features = json.loads(s["features_json"] or "[]")
    tech     = json.loads(s["tech_json"] or "[]")
    obsidian = s["obsidian_text"] or ""

    features_str = json.dumps(features, ensure_ascii=False, indent=2) if features else "없음"
    tech_str     = json.dumps(tech, ensure_ascii=False, indent=2) if tech else "없음"

    system_content = f"""당신은 Plugin Architect AI입니다. 완료된 프로젝트 결과를 사용자와 함께 검토하고 수정합니다.

## 수정 규칙 (반드시 준수)
- 피처를 수정/추가/삭제하면 응답 끝에 반드시 아래 태그로 **전체 목록**을 반환하세요:
<updated_features>
[{{"id": "F1", "name": "피처명", "description": "설명"}}]
</updated_features>

- 기술 스택을 수정/추가/삭제하면 응답 끝에 반드시 아래 태그로 **전체 목록**을 반환하세요:
<updated_tech>
["기술 추천 1", "기술 추천 2"]
</updated_tech>

- 변경이 없으면 태그 없이 대화만 하세요.
- 항상 한국어로 답하세요.

## 현재 피처 목록
{features_str}

## 현재 기술 스택
{tech_str}

## Obsidian 문서
{obsidian if obsidian else "없음"}"""

    # 이전 대화 기록 불러오기
    prev_msgs = _db(
        "SELECT role, content FROM messages WHERE thread_id = ? ORDER BY id",
        (thread_id,), fetch="all",
    )

    lc_messages = [SystemMessage(content=system_content)]
    for m in prev_msgs:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))
    lc_messages.append(HumanMessage(content=message))

    _session_msg(thread_id, "user", message, "continue")

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.4, streaming=True)
        full_response = ""
        async for chunk in llm.astream(lc_messages):
            if chunk.content:
                full_response += chunk.content
                yield sse({"type": "token", "content": chunk.content})

        _session_msg(thread_id, "assistant", full_response, "continue")
        logger.info(f"[continue-chat] response len={len(full_response)}, snippet={full_response[:120]!r}")

        # 수정된 피처/기술 스택 파싱 → DB 저장 + 프론트 패널 갱신
        updated_features = parse_updated_features(full_response)
        logger.info(f"[continue-chat] updated_features={'found' if updated_features else 'not found'}")
        if updated_features:
            _session_save_node(thread_id, "decomposer", {"features": updated_features})
            yield sse({"type": "state_update", "node": "decomposer",
                       "result": {"features": updated_features}})

        updated_tech = parse_updated_tech(full_response)
        logger.info(f"[continue-chat] updated_tech={'found' if updated_tech else 'not found'}")
        if updated_tech:
            _session_save_node(thread_id, "tech_matcher", {"tech": updated_tech})
            yield sse({"type": "state_update", "node": "tech_matcher",
                       "result": {"tech": updated_tech}})

    except Exception as e:
        logger.error(f"[continue-chat] 오류: {e}", exc_info=True)
        yield sse({"type": "error", "message": str(e)})

    yield sse({"type": "done"})


@app.post("/stream/chat/{thread_id}")
async def session_chat_endpoint(thread_id: str, req: ChatRequest):
    return StreamingResponse(
        stream_session_chat(thread_id, req.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(BASE_DIR / "frontend" / "index.html", encoding="utf-8") as f:
        return f.read()


# ── 세션 히스토리 API ─────────────────────────────────────────

@app.get("/sessions")
async def list_sessions():
    rows = _db(
        "SELECT thread_id, title, created_at, completed_at, status, "
        "features_json, tech_json, obsidian_text FROM sessions ORDER BY created_at DESC",
        fetch="all",
    )
    result = []
    for r in rows:
        result.append({
            "thread_id":     r["thread_id"],
            "title":         r["title"],
            "created_at":    r["created_at"],
            "completed_at":  r["completed_at"],
            "status":        r["status"],
            "feature_count": len(json.loads(r["features_json"] or "[]")),
            "tech_count":    len(json.loads(r["tech_json"] or "[]")),
            "has_obsidian":  bool(r["obsidian_text"]),
        })
    return result


@app.get("/sessions/{thread_id}")
async def get_session(thread_id: str):
    from fastapi import HTTPException
    s = _db(
        "SELECT thread_id, title, created_at, completed_at, status, "
        "features_json, tech_json, obsidian_text FROM sessions WHERE thread_id = ?",
        (thread_id,), fetch="one",
    )
    if not s:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    msgs = _db(
        "SELECT role, content, msg_type FROM messages WHERE thread_id = ? ORDER BY id",
        (thread_id,), fetch="all",
    )
    return {
        "thread_id":    s["thread_id"],
        "title":        s["title"],
        "created_at":   s["created_at"],
        "completed_at": s["completed_at"],
        "status":       s["status"],
        "features":     json.loads(s["features_json"] or "[]"),
        "tech":         json.loads(s["tech_json"] or "[]"),
        "obsidian":     s["obsidian_text"] or "",
        "messages":     msgs,
    }


@app.delete("/sessions/{thread_id}")
async def delete_session(thread_id: str):
    count = _db("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
    return {"ok": bool(count)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
