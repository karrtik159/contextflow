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
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from livekit.agents import AgentServer, AgentSession, JobContext, cli, room_io
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

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


# ── Agent Server ─────────────────────────────────────────────
server = AgentServer()


@server.rtc_session(agent_name="atlas-voice-agent")
async def voice_session(ctx: JobContext):
    """
    Main RTC session handler — creates an AgentSession and starts
    with the RouterAgent (fast greeting + intent classification).
    The RouterAgent will hand off to RAGAgent when needed.
    """
    session = AgentSession[SessionInfo](
        userdata=SessionInfo(
            room_name=ctx.room.name if ctx.room else "",
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
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=_get_noise_cancellation,
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


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
