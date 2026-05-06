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
| **Frontend Testing Harness** | **Gradio** | Rapid Python UI for chat demos, workflow validation, and RAG / memory debugging before a production frontend exists. |
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

### 1. Prerequisites
Ensure you have the following installed before proceeding:
- **Python:** 3.11+ (for FastAPI Backend) & 3.13 (for Voice Worker)
- **Databases:** PostgreSQL 15+ (with `pgvector`), Neo4j 5+, and a Redis Server (unless using Docker)
- **Package Manager:** `uv` (`pip install uv`)

### 2. Environment Setup
Before starting the application, you must configure the environment variables:
```bash
# Copy the example environment file
cp .env.example .env
```
> **Note:** Open the newly created `.env` file and populate it with your specific API keys (OpenAI, LiveKit) and database credentials.

### 3. Running the Application
You can start the project using one of the following methods:

#### Method A: Full Stack with Docker Compose (Recommended)
This method spins up the entire environment—including the FastAPI backend, Voice Worker, and all databases (PostgreSQL, Redis, Neo4j)—with hot-reloading enabled.

```bash
# Run the complete stack in the background
docker compose --profile dev --profile voice-dev up -d --build

# To view logs: docker compose logs -f
# To stop the stack: docker compose down
```

#### Method B: Local Setup (Manual via `uv`)
If you prefer not to use Docker for the application services, you can run the backend and the voice worker separately using two terminals. Ensure your databases are already running locally.

**Terminal 1: Start the FastAPI Backend**
```bash
uv sync  # Installs the main dependencies
uv run uvicorn app.main:app --reload
```

**Terminal 2: Start the LiveKit Voice Worker**
```bash
cd agents/voice
uv sync  # Creates an isolated environment for the Voice Agent
uv run python src/agent.py dev
```

#### Method C: Gradio Frontend Testing Harness

The repo includes an optional `demo` dependency group for a lightweight frontend testing surface with Gradio:

```bash
uv sync --extra demo
uv run python demo/gradio_app.py
```

Recommended design direction for the Gradio layer:

- Start with `gr.ChatInterface` for the fastest text-chat test harness around the FastAPI endpoints.
- Move to `gr.Blocks` once you need custom layout, feedback controls, session inspection, or side panels for memory / RAG debugging.
- Use tabs or grouped panels to separate core chat, memory inspection, and retrieval diagnostics during development.
- Treat Gradio as the **frontend testing platform**, not the long-term production UI.

Useful references:

- ChatInterface docs: https://www.gradio.app/docs/gradio/chatinterface
- Blocks guide: https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks
- API reference: https://www.gradio.app/docs

## 🧪 Testing

```bash
# FastAPI unit & integration tests
uv run pytest

# RAG quality evaluation (requires OpenAI API key)
uv run pytest tests/test_eval/

# Voice worker tests (isolated env)
cd agents/voice
uv run pytest
```

### RAGAS Evaluation

Initial RAGAS support is scaffolded for answer-level evaluation now, with a clean path to add retrieval metrics once the backend exposes retrieved contexts in an eval-friendly response.

```bash
# Requires OPENAI_API_KEY and a running FastAPI backend
uv run python scripts/run_ragas_eval.py
```

Current fixture dataset:

- `tests/test_eval/fixtures/ragas_cases.json`

Current framework entrypoints:

- `app/evals/ragas_framework.py`
- `scripts/run_ragas_eval.py`

## 🤝 Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
