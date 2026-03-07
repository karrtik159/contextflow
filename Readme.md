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
├── app/                       # Main FastAPI application
│   ├── main.py                # Entry point & lifespan events
│   ├── core/                  # Config, DB engine, security
│   ├── models/                # SQLAlchemy ORM models
│   ├── schemas/               # Pydantic V2 Create/Update/Read schemas
│   ├── api/v1/                # Versioned API routers (auth, users, chat, livekit)
│   ├── services/              # FastCRUD, vector_search, graph_search
│   └── memory/                # Mem0 integration layer
├── agents/                    # AI agents (decoupled from app)
│   ├── voice/                 # LiveKit voice transport (router, RAG)
│   └── crews/                 # CrewAI orchestration (Support, Memory)
├── tests/                     # pytest (API, agents, RAGAS eval)
├── alembic/                   # Database migrations
├── pyproject.toml             # Dependencies & tooling config
└── .env.example               # Environment variable template
```

## 🛠️ Getting Started

### Prerequisites
- Python 3.11+
- PostgreSQL 15+ with `pgvector` extension
- Neo4j 5+
- Redis Server
- API Keys: OpenAI, LiveKit, Mem0

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/karrtik159/OpenAI_Clone.git
   cd OpenAI_Clone
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

5. **Run database migrations:**
   ```bash
   alembic upgrade head
   ```

6. **Start the server:**
   ```bash
   uvicorn app.main:app --reload
   ```

7. **Open API docs:** Navigate to `http://localhost:8000/docs`

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