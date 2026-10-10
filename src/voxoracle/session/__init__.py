"""Voice-session state machine: wake -> record -> STT -> ask -> TTS.

:class:`VoiceSession` wires the wake-word detector, audio capture/VAD, cloud STT,
the DocOracle client and cloud TTS into one loop. All collaborators are injected
so a full conversation runs with fakes (no hardware, network or API key). See
``openspec/changes/add-voxoracle-core/specs/voice-session``.
"""

from __future__ import annotations

from voxoracle.session.machine import (
    Prompts,
    QuestionAnswerer,
    SessionConfig,
    SessionState,
    Speaker,
    VoiceSession,
)

__all__ = [
    "Prompts",
    "QuestionAnswerer",
    "SessionConfig",
    "SessionState",
    "Speaker",
    "VoiceSession",
]
