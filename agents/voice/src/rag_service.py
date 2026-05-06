"""
HTTP helpers for the isolated LiveKit RAG worker.
"""

import logging
import os
from typing import Any

import httpx

API_BASE_URL = os.getenv("API_BASE_URL", "http://app-dev:8000")
RAG_SERVICE_TOKEN = os.getenv("RAG_SERVICE_TOKEN", "")
CONTEXT_PREFETCH_PATH = "/api/v1/context/prefetch"
RAG_QUERY_PATH = "/api/v1/rag/query"

logger = logging.getLogger(__name__)


def get_session_user_id(session: Any) -> str:
    userdata = getattr(session, "userdata", None)
    if userdata is None:
        return "anonymous"

    user_id = getattr(userdata, "user_id", "anonymous")
    return user_id or "anonymous"


def get_chat_session_id(session: Any) -> str | None:
    userdata = getattr(session, "userdata", None)
    if userdata is None:
        return None

    chat_session_id = getattr(userdata, "chat_session_id", None)
    return chat_session_id or None


def inject_prefetched_context(turn_ctx: Any, context_data: str | None) -> None:
    if not context_data:
        return

    turn_ctx.add_message(
        role="system",
        content=f"Relevant context for the user's last message:\n{context_data}",
    )


async def _post_json(
    *,
    path: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    close_client = client is None
    active_client = client or httpx.AsyncClient(timeout=timeout_seconds)
    headers = {"X-RAG-Service-Token": RAG_SERVICE_TOKEN} if RAG_SERVICE_TOKEN else None

    try:
        response = await active_client.post(f"{API_BASE_URL}{path}", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    finally:
        if close_client:
            await active_client.aclose()


async def prefetch_context(
    *,
    query: str,
    user_id: str,
    session_id: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> str | None:
    try:
        payload: dict[str, Any] = {"query": query, "user_id": user_id}
        if session_id:
            payload["session_id"] = session_id

        response_data = await _post_json(
            path=CONTEXT_PREFETCH_PATH,
            payload=payload,
            timeout_seconds=10.0,
            client=client,
        )
        return response_data.get("context")
    except Exception as exc:
        logger.warning("Context prefetch failed: %s", exc)
        return None


async def search_knowledge_base(
    *,
    query: str,
    user_id: str,
    client: httpx.AsyncClient | None = None,
) -> str:
    try:
        response_data = await _post_json(
            path=RAG_QUERY_PATH,
            payload={"query": query, "user_id": user_id},
            timeout_seconds=30.0,
            client=client,
        )
    except httpx.HTTPStatusError as exc:
        return f"The knowledge search returned an error: {exc.response.status_code}"
    except Exception as exc:
        return f"I had trouble reaching the knowledge service: {exc}"

    return response_data.get("answer", "I couldn't find a relevant answer.")
