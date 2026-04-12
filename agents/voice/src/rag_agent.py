"""
RAG Voice Agent — handles knowledge-intensive queries.

Receives handoff from RouterAgent, calls the FastAPI /api/v1/rag/query
endpoint over HTTP for Hybrid Graph-Vector RAG, and streams the answer via TTS.

Architecture:
  Voice Worker (livekit-agents) --HTTP--> FastAPI (crewai) ---> pgvector + Neo4j + Mem0

The RAG backend now has built-in intent routing:
  - Simple chat → direct LLM (sub-second)
  - Knowledge queries → full CrewAI orchestration
So the voice agent delegates routing to the backend and only pre-fetches
context for queries the backend identifies as needing RAG.
"""
from livekit.agents import Agent, RunContext, function_tool
from livekit.agents.llm import ChatContext, ChatMessage

from rag_service import (
    get_chat_session_id,
    get_session_user_id,
    inject_prefetched_context,
    prefetch_context,
    search_knowledge_base,
)


class RAGAgent(Agent):
    """Deep Knowledge Agent — calls the FastAPI RAG endpoint over HTTP."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            instructions="""You are a knowledgeable AI specialist named Atlas.
You have access to a powerful knowledge search tool that can look up
information from documents, past conversations, and a knowledge graph.

Rules:
- Prefetched memory context may already be injected before you answer.
- If the injected context is incomplete or the user needs a grounded answer,
  use the search_knowledge tool.
- Synthesize retrieved knowledge into a concise spoken answer.
- Keep answers under 100 words — optimized for voice delivery.
- If the user wants to go back to casual chat, use transfer_to_router.
- Mention when information comes from prior conversation context versus the
  knowledge service when that distinction matters.
""",
            **kwargs,
        )

    async def on_enter(self) -> None:
        """Called when handoff from RouterAgent occurs."""
        await self.session.generate_reply(
            instructions=(
                "If the previous user turn already contains a concrete question, answer it now. "
                "Otherwise, briefly introduce yourself as the knowledge specialist and invite the user to continue."
            )
        )

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        """
        Eagerly pre-fetch context from the backend before the LLM responds.

        The FastAPI backend handles intent routing internally, so when this
        context is injected the LLM gets relevant knowledge for free.
        For simple chat, the prefetch returns minimal/no context and the
        LLM answers naturally without CrewAI overhead.
        """
        user_query = new_message.text_content
        if not user_query:
            return

        context_data = await prefetch_context(
            query=user_query,
            user_id=get_session_user_id(self.session),
            session_id=get_chat_session_id(self.session),
        )
        inject_prefetched_context(turn_ctx, context_data)

    @function_tool()
    async def search_knowledge(self, context: RunContext, query: str) -> str:
        """Search the backend RAG service for a grounded answer to the user's question."""
        return await search_knowledge_base(query=query, user_id=get_session_user_id(self.session))

    @function_tool()
    async def transfer_to_router(self, context: RunContext):
        """Transfer back to the general conversation assistant for
        casual chat, greetings, or non-knowledge questions."""
        from router_agent import RouterAgent

        return RouterAgent(chat_ctx=self.chat_ctx), "Transferring you back to general assistance."
