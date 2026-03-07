"""
LiveKit Voice Agent Worker — main entrypoint.

Starts the LiveKit AgentServer, registers the RTC session handler,
and connects to LiveKit Cloud.

Run locally:
    uv run src/agent.py dev

Run in Docker:
    docker compose --profile voice-dev up -d

LiveKit Agents SDK 1.4+ — verified against docs.livekit.io (March 2026).
"""
import json
from dataclasses import dataclass

from dotenv import load_dotenv
from livekit.agents import AgentServer, AgentSession, JobContext, cli, room_io
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.rtc import RemoteParticipant

from router_agent import RouterAgent

# Load environment variables
load_dotenv(".env.local")
load_dotenv(".env", override=False)


# ── Session State ────────────────────────────────────────────
@dataclass
class SessionInfo:
    """Custom userdata attached to each AgentSession."""

    user_id: str = "anonymous"
    room_name: str = ""
    chat_session_id: str | None = None


def _extract_chat_session_id(participant: RemoteParticipant) -> str | None:
    attributes = getattr(participant, "attributes", None) or {}
    for key in ("chat_session_id", "session_id"):
        value = attributes.get(key)
        if value:
            return value

    metadata = getattr(participant, "metadata", None)
    if not metadata:
        return None

    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError:
        return None

    for key in ("chat_session_id", "session_id"):
        value = parsed.get(key)
        if isinstance(value, str) and value:
            return value

    return None


# ── Agent Server ─────────────────────────────────────────────
server = AgentServer()


@server.rtc_session(agent_name="atlas-voice-agent")
async def voice_session(ctx: JobContext):
    """
    Main RTC session handler — creates an AgentSession and starts
    with the RouterAgent (fast greeting + intent classification).
    The RouterAgent will hand off to RAGAgent when needed.
    """
    participant = await ctx.wait_for_participant()

    session = AgentSession[SessionInfo](
        userdata=SessionInfo(
            user_id=participant.identity or "anonymous",
            room_name=ctx.room.name if ctx.room else "",
            chat_session_id=_extract_chat_session_id(participant),
        ),
        stt="deepgram/nova-3:multi",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=RouterAgent(),
        room_options=room_io.RoomOptions(
            participant_identity=participant.identity,
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=_get_noise_cancellation,
            ),
        ),
    )

    await session.generate_reply(instructions="Greet the user warmly and briefly. Offer your help.")


def _get_noise_cancellation(params):
    """Select noise cancellation based on participant kind."""
    try:
        from livekit.plugins import noise_cancellation
        from livekit.rtc import ParticipantKind

        if params.participant.kind == ParticipantKind.PARTICIPANT_KIND_SIP:
            return noise_cancellation.BVCTelephony()
        return noise_cancellation.BVC()
    except ImportError:
        return None


# ── CLI Entrypoint ───────────────────────────────────────────
if __name__ == "__main__":
    cli.run_app(server)
