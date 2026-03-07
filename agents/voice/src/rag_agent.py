"""
RAG Voice Agent — handles knowledge-intensive queries.

Receives handoff from RouterAgent, calls the FastAPI /api/v1/rag/query
endpoint over HTTP for Hybrid Graph-Vector RAG, and streams the answer via TTS.

Architecture:
  Voice Worker (livekit-agents) --HTTP--> FastAPI (crewai) ---> pgvector + Neo4j + Mem0
"""

import os

import httpx
from livekit.agents import Agent, RunContext, function_tool

# FastAPI server base URL — defaults to the Docker service name
API_BASE_URL = os.getenv("API_BASE_URL", "http://app-dev:8000")


class RAGAgent(Agent):
    """Deep Knowledge Agent — calls the FastAPI RAG endpoint over HTTP."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            instructions="""You are a knowledgeable AI specialist named Atlas.
You have access to a powerful knowledge search tool that can look up
information from documents, past conversations, and a knowledge graph.

Rules:
- When the user asks a question, use the search_knowledge tool first.
- Synthesize the tool's returned context into a concise spoken answer.
- Keep answers under 100 words — optimized for voice delivery.
- If the user wants to go back to casual chat, use transfer_to_router.
- Be confident and cite your sources when relevant.
""",
            **kwargs,
        )

    async def on_enter(self) -> None:
        """Called when handoff from RouterAgent occurs."""
        await self.session.generate_reply(
            instructions="Introduce yourself briefly as a knowledge specialist and ask what they'd like to know."
        )

    @function_tool()
    async def search_knowledge(self, context: RunContext, query: str) -> str:
        """Search the knowledge base, past conversations, and user memory
        to find relevant information for answering the user's question.
        Input: the user's question as a string."""
        user_id = "anonymous"
        if hasattr(self.session, "userdata") and self.session.userdata:
            user_id = getattr(self.session.userdata, "user_id", "anonymous")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{API_BASE_URL}/api/v1/rag/query",
                    json={"query": query, "user_id": user_id},
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("answer", "I couldn't find a relevant answer.")
        except httpx.HTTPStatusError as e:
            return f"The knowledge search returned an error: {e.response.status_code}"
        except Exception as e:
            return f"I had trouble reaching the knowledge service: {e}"

    @function_tool()
    async def transfer_to_router(self, context: RunContext):
        """Transfer back to the general conversation assistant for
        casual chat, greetings, or non-knowledge questions."""
        from router_agent import RouterAgent

        return RouterAgent(chat_ctx=self.chat_ctx), "Transferring you back to general assistance."
