"""
Router Voice Agent — the first agent a user speaks with.

Responsibilities:
  - Low-latency greeting and natural small talk.
  - Intent classification: if the user asks a knowledge-heavy question,
    trigger a handoff to the RAG Voice Agent.

NOTE: All LiveKit API usage MUST be verified against live documentation
at https://docs.livekit.io before implementation.
"""

# ── Placeholder ─────────────────────────────────────────────────────────
# This file will be implemented using the LiveKit Agents SDK.
# The router agent should:
#   1. Join a LiveKit room as a participant.
#   2. Listen for user audio via STT (Deepgram / Whisper).
#   3. Classify intent using a lightweight LLM call (fast, minimal context).
#   4. If intent == "deep_query": hand off to rag_agent.
#   5. Otherwise: respond with TTS (Cartesia / OpenAI).
#
# See: agents/crews/support_crew.py for the heavy RAG logic.
# ─────────────────────────────────────────────────────────────────────────
