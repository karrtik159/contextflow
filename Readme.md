# 🎙️ OpenAI Clone: Real-Time Voice & Deep Memory AI

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)

An advanced, high-performance open-source OpenAI Clone featuring **real-time voice interaction** and **deep memory capabilities**. By leveraging a cutting-edge **Hybrid Graph-Vector RAG (Retrieval-Augmented Generation)** architecture, this project combines the structured relationship-tracking of knowledge graphs with the high-speed semantic retrieval of vector databases to deliver unparalleled contextual awareness and ultra-low latency conversational AI.

## 🚀 Key Features

- **Hybrid Graph-Vector RAG 🧠**: Combines Vector Search (FAISS) for rapid semantic retrieval of unstructured data with Graph Data (Neo4j/Graphiti) for traceable reasoning and complex multi-hop relationship tracking.
- **Deep Personalization & Memory 🗂️**: Utilizes Mem0 or Graphiti to maintain an evolving "long-term memory," automatically constructing a comprehensive User Profile over time.
- **Real-Time Voice & Turn Detection 🔊**: Integrates LiveKit for sub-millisecond voice interactions, handling WebRTC audio streaming, native turn-detection, and graceful interruption (barge-in) handling.
- **High-Performance Async Backend ⚡**: Built on FastAPI, orchestrating parallel data fetches and LLM streams for maximum throughput and minimal blocking.

## 🏗️ Architecture & Technology Stack

Our tech stack is meticulously chosen for maximum performance, scalability, and cutting-edge AI capabilities.

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Backend Framework** | **FastAPI** | High-performance, async-first framework essential for real-time voice and streaming LLMs. |
| **Vector Database** | **FAISS** | Facebook AI Similarity Search provides the fastest semantic retrieval for massive unstructured datasets. |
| **Graph Database** | **Neo4j / Graphiti** | Essential for mapping entities and relationships (e.g., `User` -> `WorksAt` -> `Company`). |
| **Real-Time Voice** | **LiveKit** | The gold standard for WebRTC audio. Manages the room, participants, and STT/TTS pipeline seamlessly. |
| **Memory Engine** | **Mem0** | A specialized layer that automates the creation and updating of persistent user memory profiles based on chat history. |
| **Cache & Session** | **Redis** | Manages short-term context and rapid session data access natively with sub-millisecond latency. |

### The "Cutting Edge" Edge: Reranking & RRF
To ensure the highest quality context is provided to the LLM, the system implements **Reranking** after retrieval. Results from FAISS and the Graph database are combined using **Reciprocal Rank Fusion (RRF)**, prioritizing the most relevant interleaved context before inference.

### The Voice Pipeline (STT -> LLM -> TTS)
1. **Speech-to-Text (STT):** Deepgram or Whisper converts user speech to text instantly.
2. **Orchestrator (LLM):** FastAPI processes the text via the Hybrid RAG pipeline, generating a dynamically personalized response.
3. **Text-to-Speech (TTS):** Cartesia or OpenAI TTS streams the synthesized audio back with minimal perceived latency.

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- Redis Server
- Neo4j Instance (Local or AuraDB)
- API Keys for OpenAI/Anthropic, LiveKit, Deepgram, and Cartesia

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/karrtik159/OpenAI_Clone.git
   cd OpenAI_Clone
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your configuration:
   ```env
   LIVEKIT_API_KEY=your_livekit_key
   LIVEKIT_API_SECRET=your_livekit_secret
   OPENAI_API_KEY=your_openai_key
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_neo4j_password
   REDIS_URL=redis://localhost:6379
   ```

5. **Run the FastAPI Server:**
   ```bash
   uvicorn main:app --reload
   ```

## 🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.