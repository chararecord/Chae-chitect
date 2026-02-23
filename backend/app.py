"""
Plugin Architect Agent
======================
사용자의 추상적인 아이디어를 구체적인 기술 스펙으로 빌드업해주는 LangGraph 기반 에이전트.

흐름: START → decomposer_node → tech_matcher_node → obsidian_formatter_node → END
"""

from __future__ import annotations

import json
import os
import re
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

# ──────────────────────────────────────────────
# 1. State Definition
# ──────────────────────────────────────────────

class ArchitectState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    feature_breakdown: list[dict]   # decomposer_node 결과
    tech_requirements: list[str]    # tech_matcher_node 결과
    obsidian_template: str          # obsidian_formatter_node 결과
    decomposer_retries: int         # decomposer 재시도 횟수
    tech_retries: int               # tech_matcher 재시도 횟수

MAX_RETRIES = 3  # 노드당 최대 재시도 횟수 (retries <= MAX_RETRIES 조건으로 3회 허용)


# ──────────────────────────────────────────────
# 2. 헬퍼: JSON 파싱 (마크다운 코드펜스 안전하게 제거)
# ──────────────────────────────────────────────

def _parse_json(raw: str) -> list:
    """LLM 응답에서 JSON 배열을 안전하게 추출한다."""
    raw = raw.strip()
    # ```json ... ``` 또는 ``` ... ``` 형식 제거
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ──────────────────────────────────────────────
# 3. Node Implementations
# ──────────────────────────────────────────────

DECOMPOSER_SYSTEM = """\
당신은 플러그인/제품 아이디어를 독립적인 Atomic Feature로 분해하는 Plugin Architect 전문가입니다.

규칙:
- 각 피처는 서로 독립적으로 구현 가능해야 합니다 (의존성 없음).
- 중복되거나 겹치는 기능이 없어야 합니다.
- 최대 4~8개의 피처로 구성하세요.
- 반드시 아래 키를 가진 JSON 배열만 출력하세요:
    "id"          : 짧은 피처 ID (예: "F1")
    "name"        : 간결한 피처명 (한국어)
    "description" : 해당 피처가 무엇을 하는지 한 문장으로 설명 (한국어)

출력 예시 (마크다운 코드펜스 없이 JSON만):
[
  {"id": "F1", "name": "탭 콘텐츠 캡처", "description": "현재 활성화된 브라우저 탭에서 텍스트/HTML을 추출합니다."},
  {"id": "F2", "name": "AI 요약", "description": "캡처한 콘텐츠를 LLM에 전달하여 간결한 요약문을 반환합니다."}
]
"""

TECH_MATCHER_SYSTEM = """\
당신은 브라우저 확장 프로그램 및 현대 웹 애플리케이션 전문 기술 스택 어드바이저입니다.

주어진 Atomic Feature 목록을 보고 각 피처에 가장 적합한 라이브러리/도구를 추천하세요.

규칙:
- 2022년 이후 활발히 유지되는 최신 라이브러리를 우선하세요.
- 관련 있다면 Chrome Extension Manifest V3, TypeScript, React, Vite를 고려하세요.
- 피처 하나당 추천 하나만 작성하세요. (패키지명 + 이유를 한국어로 간결하게)
- 반드시 문자열 JSON 배열만 출력하세요 (마크다운 코드펜스 없이).

출력 예시:
[
  "chrome.scripting.executeScript API — MV3 내장 API로 탭 콘텐츠 추출에 최적",
  "openai npm 패키지 (v4+) — GPT 요약 엔드포인트 호출을 위한 경량 클라이언트"
]
"""

OBSIDIAN_FORMATTER_SYSTEM = """\
당신은 Obsidian 문서 디자이너입니다. 주어진 피처 분석과 기술 스택 추천을 \
Obsidian에 바로 붙여넣을 수 있는 완성도 높은 한국어 마크다운 문서로 변환하세요.

요구사항:
1. **H1 제목**: 플러그인/제품 이름 (컨텍스트에서 추론, 한국어).
2. **요약 callout**: `> [!info] 개요` — 1~2문장 한국어 설명.
3. **피처 섹션** (`## ✨ 주요 기능`):
   - 피처별 Obsidian callout: `> [!example] F1 · 피처명`
   - callout 아래: 한국어 설명 + 매칭된 기술 스택 (이탤릭체).
4. **기술 스택 섹션** (`## 🛠 기술 스택`):
   - 추천 라이브러리 전체 bullet 리스트 (한국어 설명 포함).
5. **아키텍처 섹션** (`## 🗺 아키텍처`):
   - Mermaid flowchart (`graph LR`)으로 전체 데이터 흐름 시각화:
     사용자 → 피처 → 기술 레이어.

마크다운 본문만 출력하세요. 앞뒤로 어떠한 설명도 추가하지 마세요.
"""


def decomposer_node(state: ArchitectState) -> dict:
    """사용자 아이디어 → Atomic Features 리스트"""
    retries = state.get("decomposer_retries", 0)
    # 재시도 시 temperature 소폭 상승 → 다양한 결과 유도
    llm = ChatOpenAI(model="gpt-4o", temperature=min(0.2 + retries * 0.15, 0.7))

    idea = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            idea = msg.content
            break

    retry_hint = ""
    if retries > 0:
        prev_count = len(state.get("feature_breakdown", []))
        retry_hint = (
            f"\n\n[품질 검증 재시도 {retries}회] "
            f"이전 시도에서 {prev_count}개의 피처만 생성됐습니다. "
            "반드시 4개 이상의 독립적이고 구체적인 피처를 생성하세요. "
            "각 피처에 id, name, description이 모두 포함돼야 합니다."
        )

    response = llm.invoke([
        SystemMessage(content=DECOMPOSER_SYSTEM),
        HumanMessage(content=f"아이디어: {idea}{retry_hint}"),
    ])

    try:
        feature_breakdown: list[dict] = _parse_json(response.content)
    except (json.JSONDecodeError, ValueError):
        feature_breakdown = []

    return {
        "feature_breakdown": feature_breakdown,
        "decomposer_retries": retries + 1,
        "messages": [AIMessage(content=f"[Decomposer] {len(feature_breakdown)}개의 Atomic Feature로 분해 완료.")],
    }


def tech_matcher_node(state: ArchitectState) -> dict:
    """Atomic Features → 기술 스택 추천 리스트"""
    retries = state.get("tech_retries", 0)
    llm = ChatOpenAI(model="gpt-4o", temperature=min(0.2 + retries * 0.15, 0.7))
    features = state.get("feature_breakdown", [])

    features_text = json.dumps(features, ensure_ascii=False, indent=2)

    retry_hint = ""
    if retries > 0:
        prev_count = len(state.get("tech_requirements", []))
        retry_hint = (
            f"\n\n[품질 검증 재시도 {retries}회] "
            f"이전 시도에서 {prev_count}개의 기술만 추천됐으나 피처가 {len(features)}개입니다. "
            f"반드시 피처 수({len(features)})와 동일한 수의 기술 추천을 반환하세요."
        )

    response = llm.invoke([
        SystemMessage(content=TECH_MATCHER_SYSTEM),
        HumanMessage(content=f"Features:\n{features_text}{retry_hint}"),
    ])

    try:
        tech_requirements: list[str] = _parse_json(response.content)
    except (json.JSONDecodeError, ValueError):
        tech_requirements = []

    return {
        "tech_requirements": tech_requirements,
        "tech_retries": retries + 1,
        "messages": [AIMessage(content=f"[TechMatcher] {len(tech_requirements)}개의 기술 스택 매칭 완료.")],
    }


def obsidian_formatter_node(state: ArchitectState) -> dict:
    """Features + Tech Stack → Obsidian 마크다운 문서"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    features = state.get("feature_breakdown", [])
    tech = state.get("tech_requirements", [])

    combined = (
        f"## Feature Breakdown\n{json.dumps(features, ensure_ascii=False, indent=2)}\n\n"
        f"## Tech Requirements\n{json.dumps(tech, ensure_ascii=False, indent=2)}"
    )

    response = llm.invoke([
        SystemMessage(content=OBSIDIAN_FORMATTER_SYSTEM),
        HumanMessage(content=combined),
    ])

    obsidian_template = response.content.strip()

    return {
        "obsidian_template": obsidian_template,
        "messages": [AIMessage(content="[Formatter] Obsidian 마크다운 생성 완료.")],
    }


# ──────────────────────────────────────────────
# 4. Conditional Edge Routing
# ──────────────────────────────────────────────

def route_decomposer(state: ArchitectState) -> str:
    """품질 검증: 피처가 3개 미만이거나 필수 필드가 빠지면 재시도."""
    features  = state.get("feature_breakdown", [])
    retries   = state.get("decomposer_retries", 0)

    quality_ok = (
        len(features) >= 3
        and all(f.get("id") and f.get("name") and f.get("description") for f in features)
    )

    if not quality_ok and retries <= MAX_RETRIES:
        return "decomposer"   # ← 루프백 (재시도)
    return "tech_matcher"     # → 다음 노드


def route_tech_matcher(state: ArchitectState) -> str:
    """품질 검증: 기술 추천 수가 피처 수보다 적으면 재시도."""
    features  = state.get("feature_breakdown", [])
    tech      = state.get("tech_requirements", [])
    retries   = state.get("tech_retries", 0)

    quality_ok = len(tech) >= len(features)

    if not quality_ok and retries <= MAX_RETRIES:
        return "tech_matcher"        # ← 루프백 (재시도)
    return "obsidian_formatter"      # → 다음 노드


# ──────────────────────────────────────────────
# 5. Graph Construction
# ──────────────────────────────────────────────

def build_architect_graph():
    """Plugin Architect 그래프를 빌드하고 컴파일한다."""
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()

    graph = StateGraph(ArchitectState)

    graph.add_node("decomposer", decomposer_node)
    graph.add_node("tech_matcher", tech_matcher_node)
    graph.add_node("obsidian_formatter", obsidian_formatter_node)

    graph.add_edge(START, "decomposer")

    # Conditional Edges: 품질 통과 시 다음 노드, 실패 시 자기 자신으로 루프백
    graph.add_conditional_edges(
        "decomposer",
        route_decomposer,
        {"decomposer": "decomposer", "tech_matcher": "tech_matcher"},
    )
    graph.add_conditional_edges(
        "tech_matcher",
        route_tech_matcher,
        {"tech_matcher": "tech_matcher", "obsidian_formatter": "obsidian_formatter"},
    )

    graph.add_edge("obsidian_formatter", END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_after=["decomposer", "tech_matcher"],
    )


# ──────────────────────────────────────────────
# 6. Mermaid 시각화 유틸리티
# ──────────────────────────────────────────────

def print_mermaid(app) -> None:
    """컴파일된 그래프의 Mermaid 다이어그램을 출력한다."""
    try:
        mermaid_code = app.get_graph().draw_mermaid()
        print("\n─── Graph Mermaid Diagram ───────────────────────")
        print(mermaid_code)
        print("─────────────────────────────────────────────────\n")
    except Exception as exc:
        print(f"[Mermaid] 시각화 불가: {exc}")


# ──────────────────────────────────────────────
# 7. Sample Run
# ──────────────────────────────────────────────

def run(idea: str) -> ArchitectState:
    """
    샘플 실행 함수.

    Args:
        idea: 구체화할 플러그인/제품 아이디어 문자열

    Returns:
        최종 ArchitectState (feature_breakdown, tech_requirements, obsidian_template 포함)
    """
    app = build_architect_graph()
    print_mermaid(app)

    initial_state: ArchitectState = {
        "messages": [HumanMessage(content=idea)],
        "feature_breakdown": [],
        "tech_requirements": [],
        "obsidian_template": "",
    }

    print(f"[START] 아이디어 입력: {idea}\n")

    result: ArchitectState = app.invoke(initial_state)

    print("\n═══════════════════════════════════════════════")
    print("  Plugin Architect Agent — 결과 요약")
    print("═══════════════════════════════════════════════")

    print("\n[1] Feature Breakdown")
    for feat in result["feature_breakdown"]:
        print(f"  • {feat.get('id')} | {feat.get('name')}: {feat.get('description')}")

    print("\n[2] Tech Requirements")
    for i, tech in enumerate(result["tech_requirements"], 1):
        print(f"  {i}. {tech}")

    print("\n[3] Obsidian Template (복사해서 붙여넣기)")
    print("─" * 50)
    print(result["obsidian_template"])
    print("─" * 50)

    return result


if __name__ == "__main__":
    SAMPLE_IDEA = "브라우저 대화 요약 확장 프로그램"
    run(SAMPLE_IDEA)
