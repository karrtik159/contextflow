"""
LiveKit Agent Worker — entrypoint for the voice AI agent server.

Starts the LiveKit AgentServer, registers the RTC session handler,
and connects to the LiveKit Cloud (or self-hosted) server.

Run with:
    python -m agents.voice.worker dev

LiveKit Agents SDK 1.4+ — verified against docs.livekit.io (March 2026).
"""

from dataclasses import dataclass

from dotenv import load_dotenv
from livekit.agents import AgentServer, AgentSession, JobContext, cli, room_io
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.rtc import ParticipantKind

from agents.voice.router_agent import RouterAgent

# Load environment variables (.env or .env.local)
load_dotenv(".env")
load_dotenv(".env.local", override=True)


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
        # STT → LLM → TTS pipeline (LiveKit Inference model strings)
        stt="deepgram/nova-3:multi",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        # Voice Activity Detection + Turn Detection
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

    # Generate the initial greeting
    await session.generate_reply(instructions="Greet the user and offer your assistance.")


def _get_noise_cancellation(params):
    """Select noise cancellation based on participant kind (SIP vs browser)."""
    try:
        from livekit.plugins import noise_cancellation

        if params.participant.kind == ParticipantKind.PARTICIPANT_KIND_SIP:
            return noise_cancellation.BVCTelephony()
        return noise_cancellation.BVC()
    except ImportError:
        return None


# ── CLI Entrypoint ───────────────────────────────────────────
if __name__ == "__main__":
    cli.run_app(server)
