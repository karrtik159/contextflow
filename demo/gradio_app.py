"""
Gradio frontend testing harness for the ContextFlow backend.

This app is intentionally focused on development workflows:
- RAG chat against the FastAPI backend
- Memory add/search/list debugging
- LiveKit token generation for voice testing
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, parse, request

import gradio as gr

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_USER_ID = os.getenv("GRADIO_DEMO_USER_ID", "demo-user")
DEFAULT_ROOM_NAME = os.getenv("GRADIO_DEMO_ROOM", "atlas-room")

APP_CSS = """
:root {
  --atlas-ink: #0f172a;
  --atlas-steel: #334155;
  --atlas-sand: #f8f5ef;
  --atlas-gold: #f59e0b;
  --atlas-rust: #c2410c;
  --atlas-cloud: #fffdf8;
}

.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(245, 158, 11, 0.14), transparent 28%),
    linear-gradient(180deg, #fffdf8 0%, #f7f2e8 100%);
}

.atlas-shell {
  max-width: 1240px;
  margin: 0 auto;
}

.atlas-hero {
  padding: 24px 28px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(255, 247, 237, 0.96));
  border-radius: 24px;
  box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
}

.atlas-kicker {
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--atlas-rust);
  font-weight: 700;
  font-size: 0.8rem;
}

.atlas-title {
  margin: 0.35rem 0 0;
  color: var(--atlas-ink);
  font-size: 2.4rem;
  line-height: 1.05;
}

.atlas-copy {
  color: var(--atlas-steel);
  margin: 0.75rem 0 0;
  max-width: 60rem;
}

.atlas-card {
  border: 1px solid rgba(15, 23, 42, 0.08);
  background: rgba(255, 255, 255, 0.92);
  border-radius: 20px;
  padding: 16px;
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
}
"""


def _pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = value.strip()
    return cleaned or None


def _request_json(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    target = f"{API_BASE_URL}{path}"
    data = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(target, data=data, method=method.upper(), headers=headers)
    try:
        with request.urlopen(req, timeout=45) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body) if body else None
            return {"ok": True, "status_code": response.status, "data": parsed}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            parsed = json.loads(body) if body else {"detail": exc.reason}
        except json.JSONDecodeError:
            parsed = {"detail": body or exc.reason}
        return {"ok": False, "status_code": exc.code, "error": parsed}
    except error.URLError as exc:
        return {
            "ok": False,
            "status_code": None,
            "error": {"detail": f"Could not reach {target}: {exc.reason}"},
        }


def send_rag_message(
    message: str,
    history: list[dict[str, str]] | None,
    user_id: str,
) -> tuple[list[dict[str, str]], str, str]:
    clean_message = message.strip()
    if not clean_message:
        return history or [], "", _pretty_json({"detail": "Enter a message before sending."})

    response = _request_json(
        "POST",
        "/api/v1/rag/query",
        {
            "query": clean_message,
            "user_id": _clean_optional(user_id),
        },
    )
    next_history = list(history or [])
    next_history.append({"role": "user", "content": clean_message})

    if response["ok"]:
        payload = response["data"] or {}
        answer = payload.get("answer", "The backend returned no answer.")
    else:
        payload = response["error"]
        answer = f"Request failed: {payload.get('detail', 'Unknown error')}"

    next_history.append({"role": "assistant", "content": answer})
    return next_history, "", _pretty_json(response)


def add_memory(content: str, user_id: str, metadata_text: str) -> tuple[str, str]:
    clean_content = content.strip()
    if not clean_content:
        message = {"detail": "Enter memory content before storing."}
        return _pretty_json(message), _pretty_json(message)

    metadata = None
    if metadata_text.strip():
        try:
            metadata = json.loads(metadata_text)
        except json.JSONDecodeError as exc:
            message = {"detail": f"Metadata must be valid JSON: {exc.msg}"}
            return _pretty_json(message), _pretty_json(message)

    response = _request_json(
        "POST",
        "/api/v1/memories/add",
        {
            "content": clean_content,
            "user_id": user_id.strip(),
            "metadata": metadata,
        },
    )
    return _pretty_json(response.get("data") if response["ok"] else response["error"]), _pretty_json(response)


def search_memory(query: str, user_id: str, limit: int) -> tuple[str, str]:
    response = _request_json(
        "POST",
        "/api/v1/memories/search",
        {
            "query": query.strip(),
            "user_id": user_id.strip(),
            "limit": int(limit),
        },
    )
    return _pretty_json(response.get("data") if response["ok"] else response["error"]), _pretty_json(response)


def list_memories(user_id: str) -> tuple[str, str]:
    safe_user_id = parse.quote(user_id.strip())
    response = _request_json("GET", f"/api/v1/memories/{safe_user_id}")
    return _pretty_json(response.get("data") if response["ok"] else response["error"]), _pretty_json(response)


def create_livekit_token(user_id: str, room_name: str, chat_session_id: str) -> tuple[str, str]:
    response = _request_json(
        "POST",
        "/api/v1/livekit/token",
        {
            "user_id": user_id.strip(),
            "room_name": room_name.strip(),
            "chat_session_id": _clean_optional(chat_session_id),
        },
    )

    if response["ok"]:
        payload = response["data"] or {}
        token = payload.get("token", "")
        return token, _pretty_json(response)

    return "", _pretty_json(response)


def clear_chat() -> tuple[list, str, str]:
    return [], "", _pretty_json({"status": "cleared"})


with gr.Blocks(
    title="Atlas Frontend Test Bench",
    fill_width=True,
) as demo:
    with gr.Column(elem_classes=["atlas-shell"]):
        gr.HTML(
            f"""
            <section class="atlas-hero">
              <div class="atlas-kicker">Frontend Testing Platform</div>
              <h1 class="atlas-title">Atlas Gradio Control Room</h1>
              <p class="atlas-copy">
                Exercise the FastAPI RAG path, inspect memory operations, and generate LiveKit tokens
                without waiting on a production frontend. Current backend target:
                <strong>{API_BASE_URL}</strong>
              </p>
            </section>
            """
        )

        with gr.Tabs():
            with gr.TabItem("RAG Chat"):
                with gr.Row():
                    with gr.Column(scale=3, elem_classes=["atlas-card"]):
                        chat_user_id = gr.Textbox(
                            label="User ID",
                            value=DEFAULT_USER_ID,
                            placeholder="UUID or logical user key",
                        )
                        chatbot = gr.Chatbot(
                            label="Atlas Chat",
                            placeholder="Ask the backend a question. Responses come from /api/v1/rag/query.",
                            height=480,
                        )
                        rag_input = gr.Textbox(
                            label="Message",
                            placeholder="How does the voice worker reach the backend?",
                        )
                        with gr.Row():
                            send_button = gr.Button("Send", variant="primary")
                            clear_button = gr.Button("Clear")
                    with gr.Column(scale=2, elem_classes=["atlas-card"]):
                        gr.Markdown(
                            "### Request Trace\n"
                            "Use this pane to inspect the exact backend payload and response body."
                        )
                        rag_trace = gr.Code(
                            label="RAG Response",
                            language="json",
                            value=_pretty_json({"status": "idle"}),
                            lines=24,
                        )

                send_button.click(
                    send_rag_message,
                    inputs=[rag_input, chatbot, chat_user_id],
                    outputs=[chatbot, rag_input, rag_trace],
                )
                rag_input.submit(
                    send_rag_message,
                    inputs=[rag_input, chatbot, chat_user_id],
                    outputs=[chatbot, rag_input, rag_trace],
                )
                clear_button.click(clear_chat, outputs=[chatbot, rag_input, rag_trace])

            with gr.TabItem("Memory Console"):
                with gr.Row():
                    memory_user_id = gr.Textbox(label="User ID", value=DEFAULT_USER_ID)
                    memory_limit = gr.Slider(label="Search Limit", minimum=1, maximum=20, step=1, value=5)

                with gr.Row():
                    with gr.Column(elem_classes=["atlas-card"]):
                        gr.Markdown("### Add Memory")
                        memory_content = gr.Textbox(
                            label="Content",
                            lines=5,
                            placeholder="Atlas prefers concise deployment checklists.",
                        )
                        memory_metadata = gr.Code(
                            label="Metadata JSON",
                            language="json",
                            value='{\n  "source": "gradio-demo"\n}',
                            lines=6,
                        )
                        memory_add_button = gr.Button("Store Memory", variant="primary")

                    with gr.Column(elem_classes=["atlas-card"]):
                        gr.Markdown("### Search / List")
                        memory_query = gr.Textbox(
                            label="Search Query",
                            placeholder="deployment checklist",
                        )
                        with gr.Row():
                            memory_search_button = gr.Button("Search Memories")
                            memory_list_button = gr.Button("List All")

                with gr.Row():
                    memory_result = gr.Code(label="Memory Result", language="json", lines=16, value="{}")
                    memory_trace = gr.Code(label="HTTP Trace", language="json", lines=16, value="{}")

                memory_add_button.click(
                    add_memory,
                    inputs=[memory_content, memory_user_id, memory_metadata],
                    outputs=[memory_result, memory_trace],
                )
                memory_search_button.click(
                    search_memory,
                    inputs=[memory_query, memory_user_id, memory_limit],
                    outputs=[memory_result, memory_trace],
                )
                memory_list_button.click(
                    list_memories,
                    inputs=[memory_user_id],
                    outputs=[memory_result, memory_trace],
                )

            with gr.TabItem("LiveKit Token"):
                with gr.Row():
                    with gr.Column(scale=2, elem_classes=["atlas-card"]):
                        token_user_id = gr.Textbox(label="User ID", value=DEFAULT_USER_ID)
                        token_room_name = gr.Textbox(label="Room Name", value=DEFAULT_ROOM_NAME)
                        token_session_id = gr.Textbox(
                            label="Chat Session ID",
                            placeholder="Optional UUID used by the voice worker for scoped RAG prefetch",
                        )
                        token_button = gr.Button("Create Token", variant="primary")
                    with gr.Column(scale=3, elem_classes=["atlas-card"]):
                        livekit_token = gr.Textbox(label="JWT", lines=8, placeholder="Generated token appears here")
                        livekit_trace = gr.Code(
                            label="Token Response",
                            language="json",
                            lines=18,
                            value=_pretty_json({"status": "idle"}),
                        )

                token_button.click(
                    create_livekit_token,
                    inputs=[token_user_id, token_room_name, token_session_id],
                    outputs=[livekit_token, livekit_trace],
                )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="amber",
            secondary_hue="orange",
            neutral_hue="slate",
        ),
        css=APP_CSS,
    )
