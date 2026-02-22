# Chae-chitect

**LangGraph 기반 플러그인 아키텍트 에이전트**

아이디어를 입력하면 LangGraph 파이프라인이 자동으로 원자적 피처를 분해하고, 기술 스택을 추천하며, Obsidian 마크다운 문서를 생성합니다.
각 단계마다 Human-in-the-loop 검토가 가능하고, 품질 기준 미달 시 자동 재시도(최대 3회)를 수행합니다.

---

## 주요 기능

| 기능 | 설명 |
|---|---|
| **StateGraph 파이프라인** | decomposer → tech_matcher → obsidian_formatter 3단계 노드 |
| **Human-in-the-loop** | `interrupt_after`로 각 노드 완료 후 사람이 검토·수정 가능 |
| **Cycles & Conditional Edges** | 품질 기준 미달 시 최대 3회 자동 재시도 루프 |
| **Persistence & Checkpoints** | `MemorySaver`로 실행 단계별 State 스냅샷 저장 |
| **실시간 스트리밍** | SSE(Server-Sent Events)로 LLM 토큰 및 노드 상태 실시간 전달 |
| **LangGraph 시각화 패널** | 그래프 흐름, State, Event Log를 우측 패널에 실시간 표시 |

---

## 프로젝트 구조

```
Chae-chitect/
├── backend/
│   ├── app.py        # LangGraph StateGraph 정의 (노드, 라우터, 그래프 컴파일)
│   └── server.py     # FastAPI 서버 (SSE 스트리밍, Human-in-the-loop API)
├── frontend/
│   └── index.html    # 커스텀 채팅 UI (두 패널 레이아웃 + LangGraph 시각화)
├── .env.example      # 환경 변수 예시
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 시작하기

### 1. 환경 설정

```bash
# Python 3.10 이상 필요
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. API 키 설정

```bash
cp .env.example .env
# .env 파일을 열고 OPENAI_API_KEY 입력
```

### 3. 서버 실행

```bash
# 프로젝트 루트에서 실행
uvicorn backend.server:app --reload --port 8000
```

### 4. 브라우저 접속

```
http://localhost:8000
```

---

## LangGraph 개념 설명

### StateGraph
모든 노드가 공유하는 `ArchitectState`(TypedDict)를 중심으로 그래프를 구성합니다.
각 노드는 State를 읽고, 처리 결과를 State에 업데이트합니다.

### Checkpointer (Persistence)
`MemorySaver`를 통해 각 실행 단계의 State를 스냅샷으로 저장합니다.
`thread_id`로 대화를 식별하며, Human-in-the-loop 재개 시 이전 State를 복원합니다.

### interrupt_after (Human-in-the-loop)
`decomposer`, `tech_matcher` 노드 완료 후 자동으로 그래프를 일시 중단합니다.
사용자가 결과를 검토·수정하고 재개(resume)하면 그 시점부터 다시 실행됩니다.

### Conditional Edges (Cycles)
각 노드 완료 후 품질 기준을 검사하는 라우터 함수가 실행됩니다.
기준 미달이면 해당 노드를 최대 3회까지 재시도(루프)하고, 통과하면 다음 노드로 진행합니다.

---

## 기술 스택

- **LangGraph** `>=0.2.0` — 에이전트 파이프라인
- **LangChain** `>=0.3.0` — LLM 추상화
- **FastAPI** — 비동기 웹 서버
- **OpenAI GPT** — LLM
- **SSE** — 실시간 스트리밍
