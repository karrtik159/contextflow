"""
Router Voice Agent — the first agent a user speaks with.

Low-latency greeting, natural small talk, and intent classification.
If the user asks a knowledge-heavy question, hands off to RAGAgent.
"""

from livekit.agents import Agent, RunContext, function_tool


class RouterAgent(Agent):
    """Greeting & Router — minimal context for maximum speed."""

    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly AI voice assistant named Atlas.
Your job is to greet the user, handle casual small talk, and detect when
the user needs a detailed, knowledge-heavy answer.

Rules:
- Be concise — keep responses under 30 words for voice.
- If the user asks a question that requires searching a knowledge base,
  database, or past conversations, use the transfer_to_rag tool.
- For simple greetings, jokes, or casual chat, respond directly.
- Never say "I don't know" — instead, transfer to the specialist.
""",
        )

    async def on_enter(self) -> None:
        """Called when this agent becomes active."""
        await self.session.generate_reply(instructions="Greet the user warmly and briefly. Offer your help.")

    @function_tool()
    async def transfer_to_rag(self, context: RunContext):
        """Transfer the conversation to a Deep Knowledge specialist who can
        search the knowledge base, past conversations, and user memory to
        give a thorough, accurate answer."""
        from rag_agent import RAGAgent

        return RAGAgent(chat_ctx=self.chat_ctx), "Let me connect you with our knowledge specialist."
