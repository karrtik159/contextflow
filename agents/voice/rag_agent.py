"""
RAG Voice Agent — handles knowledge-intensive queries.

Responsibilities:
  - Receives a handoff from the Router Agent when a deep query is detected.
  - Invokes the CrewAI Support Crew to gather context from pgvector + Neo4j.
  - Streams the synthesized answer back to the user via TTS.

NOTE: All LiveKit API usage MUST be verified against live documentation
at https://docs.livekit.io before implementation.
"""

# ── Placeholder ─────────────────────────────────────────────────────────
# This file will be implemented using the LiveKit Agents SDK.
# The RAG agent should:
#   1. Accept a handoff from the router agent (with conversation context).
#   2. Call agents.crews.support_crew.SupportCrew.kickoff() to run
#      the full Hybrid RAG pipeline.
#   3. Stream the LLM response back to the user via TTS.
#   4. After the response, optionally trigger the Memory Crew in the
#      background to update long-term user memory.
# ─────────────────────────────────────────────────────────────────────────
