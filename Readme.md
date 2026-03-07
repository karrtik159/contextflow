# 🎙️ OpenAI Clone: Real-Time Voice & Deep Memory AI

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135%2B-green.svg)

An advanced, high-performance open-source OpenAI Clone featuring **real-time voice interaction** and **deep memory capabilities**. By leveraging a cutting-edge **Hybrid Graph-Vector RAG (Retrieval-Augmented Generation)** architecture, this project combines the structured relationship-tracking of knowledge graphs with the high-speed semantic retrieval of vector databases to deliver unparalleled contextual awareness and ultra-low latency conversational AI.

## 🚀 Key Features

- **Hybrid Graph-Vector RAG 🧠** — Combines PostgreSQL (`pgvector`) for rapid semantic retrieval with Neo4j for traceable, multi-hop reasoning via Reciprocal Rank Fusion (RRF).
- **Deep Personalization & Memory 🗂️** — Mem0 maintains an evolving long-term memory profile per user, automatically extracting entities and preferences from conversations.
- **Multi-Agent Orchestration 🤖** — CrewAI powers modular, role-based AI crews (Support Crew for RAG, Memory Crew for graph updates) decoupled from the voice transport layer.
- **Real-Time Voice & Turn Detection 🔊** — LiveKit handles sub-millisecond WebRTC voice streaming, native turn-detection, and graceful barge-in interruption.
- **High-Performance Async Backend ⚡** — FastAPI with async SQLAlchemy 2.0, FastCRUD for zero-boilerplate data access, and Pydantic V2 for strict schema validation.
- **Evaluation & Traceability 📊** — LangSmith / OpenTelemetry for execution tracing, RAGAS for RAG quality evaluation (Faithfulness, Context Precision).

## 🏗️ Architecture & Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Backend Framework** | **FastAPI** | Async-first framework for real-time voice and streaming LLMs. |
| **Database (Relational)** | **PostgreSQL + SQLAlchemy 2.0** | Async ORM with connection pooling via `asyncpg`. |
| **Database (Vector)** | **PostgreSQL + pgvector** | Semantic similarity search — ACID-compliant, co-located with relational data. |
| **Graph Database** | **Neo4j** | Multi-hop relationship mapping for deep memory and reasoning. |
| **CRUD Layer** | **FastCRUD + Pydantic V2** | Schema-driven, zero-boilerplate CRUD with Alembic migrations. |
| **AI Orchestration** | **CrewAI** | Role-based multi-agent crews (Support, Memory) with YAML config. |
| **Memory Engine** | **Mem0** | Automated long-term user memory backed by pgvector + Neo4j. |
| **Real-Time Voice** | **LiveKit** | WebRTC audio, STT/TTS pipeline, turn detection, barge-in handling. |
| **Cache & Session** | **Redis** | Sub-millisecond session state and short-term context. |
| **Evaluation** | **RAGAS + LangSmith** | RAG quality metrics and full execution trace observability. |

## 📁 Project Structure

```text
OpenAI_Clone/
├── app/                       # Main FastAPI application (CrewAI, DBs, Auth)
├── agents/
│   ├── voice/                 # 🎙️ LiveKit Voice Worker (Standalone uv project)
│   │   ├── src/agent.py       # Voice entrypoint
│   │   └── pyproject.toml     # Isolated dependencies (livekit-agents, httpx)
│   └── crews/                 # CrewAI orchestration (Support, Memory)
├── tests/                     # pytest (API, agents, RAGAS eval)
├── alembic/                   # Database migrations
├── docker-compose.yml         # Full stack deployment (FastAPI, Voice Worker, DBs)
└── pyproject.toml             # FastAPI Main Dependencies
```

### 🧱 Agent Architecture (Separation of Concerns)

To avoid dependency conflicts (specifically `opentelemetry-sdk` mismatches between `crewai` and `livekit-agents`), the Voice Agent runs as a **completely isolated process**.

1. **Voice Worker (`agents/voice`)**: Connects to LiveKit Cloud, streams STT/TTS, handles turn detection.
2. **FastAPI RAG Endpoint**: Runs CrewAI, connects to Postgres + Neo4j.
3. **Integration**: The Voice Worker calls `POST /api/v1/rag/query` over HTTP to fetch knowledge, keeping environments clean.

## 🛠️ Getting Started

### Prerequisites
- Python 3.11+ (FastAPI) & Python 3.13 (Voice Worker)
- PostgreSQL 15+ with `pgvector`
- Neo4j 5+ & Redis Server
- `uv` package manager (`pip install uv`)

### Option A: Full Stack with Docker (Recommended)

Start the entire environment (FastAPI, Voice Worker, Postgres, Redis, Neo4j) with hot-reloading:

```bash
cp .env.example .env
# Edit .env with your OpenAI, LiveKit, and DB credentials
docker compose --profile dev --profile voice-dev up -d --build
```

### Option B: Local Setup (Two Terminals)

**Terminal 1: FastAPI Backend**
```bash
uv sync  # Installs main dependencies
uv run uvicorn app.main:app --reload
```

**Terminal 2: LiveKit Voice Worker**
```bash
cd agents/voice
uv sync  # Creates isolated environment for LiveKit agents
uv run python src/agent.py dev
```

## 🧪 Testing

```bash
# Unit & integration tests
pytest

# RAG quality evaluation (requires OpenAI API key)
pytest tests/test_eval/
```

## 🤝 Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.