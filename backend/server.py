"""
Chae-chitect — FastAPI + SSE 백엔드 (Human-in-the-Loop Review Chat)
"""

import asyncio
import json
import re
import threading
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

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

            await evt.wait()
            _continue_events.pop(thread_id, None)
            _review_history.pop(thread_id, None)
            _review_node.pop(thread_id, None)

            yield sse({"type": "review_end"})
            inputs = None
        else:
            break

    yield sse({"type": "done"})


# ── 리뷰 대화 ────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class ReviewRequest(BaseModel):
    message: str


async def stream_review_response(thread_id: str, user_message: str):
    import logging
    logger = logging.getLogger("review")

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

        llm = ChatOpenAI(model="gpt-4o", temperature=0.3, streaming=True)
        full_response = ""

        async for chunk in llm.astream(lc_messages):
            if chunk.content:
                full_response += chunk.content
                yield sse({"type": "token", "content": chunk.content})

        logger.info(f"[review] LLM responded {len(full_response)} chars")
        history.append({"role": "assistant", "content": full_response})

        # 상태 업데이트 파싱
        if node == "decomposer":
            updated = parse_updated_features(full_response)
            if updated:
                architect_app.update_state(config, {"feature_breakdown": updated})
                yield sse({"type": "state_update", "node": node, "result": {"features": updated}})

        elif node == "tech_matcher":
            updated = parse_updated_tech(full_response)
            if updated:
                architect_app.update_state(config, {"tech_requirements": updated})
                yield sse({"type": "state_update", "node": node, "result": {"tech": updated}})

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


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(BASE_DIR / "frontend" / "index.html", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
