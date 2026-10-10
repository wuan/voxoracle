"""Build a ready-to-run :class:`VoiceSession` from :class:`Settings`.

This is the one place that turns configuration into concrete, device-backed
collaborators (PortAudio capture/playback, the openWakeWord model, the Mistral
client and the DocOracle client). Hardware and provider imports are deferred to
this module so the rest of ``voxoracle.session`` stays importable without
PortAudio, openWakeWord or a network.
"""

from __future__ import annotations

from dataclasses import dataclass

from voxoracle.audio.protocols import AudioInput, AudioOutput
from voxoracle.config import Settings, resolve_mistral_api_key, resolve_models_dir
from voxoracle.docoracle.client import DocOracleClient
from voxoracle.mistral.client import MistralAudioClient
from voxoracle.session.machine import SessionConfig, VoiceSession
from voxoracle.stt.protocol import Transcriber
from voxoracle.tts.protocol import Synthesizer


class ConfigurationError(Exception):
    """Raised when the device components cannot be built from the settings."""


@dataclass
class SessionComponents:
    """The live components backing a :class:`VoiceSession`, for clean shutdown."""

    session: VoiceSession
    audio_input: AudioInput
    audio_output: AudioOutput
    mistral_client: MistralAudioClient | None = None
    docoracle_client: DocOracleClient | None = None

    def close(self) -> None:
        """Release device streams. Async clients are closed by :meth:`aclose`."""
        for close in (
            getattr(self.audio_input, "close", None),
            getattr(self.audio_output, "close", None),
        ):
            if callable(close):
                close()

    async def aclose(self) -> None:
        """Close device streams first, then the HTTP clients."""
        self.close()
        if self.mistral_client is not None:
            await self.mistral_client.aclose()
        if self.docoracle_client is not None:
            await self.docoracle_client.aclose()


def build_session(settings: Settings) -> SessionComponents:
    """Resolve devices, models and clients from ``settings`` into a session.

    Raises :class:`ConfigurationError` when a required device, model or
    credential is missing, so ``voxoracle run`` can fail with a clear message
    instead of a traceback.
    """
    api_key = resolve_mistral_api_key(settings)
    if not api_key:
        raise ConfigurationError(
            "no Mistral API key configured (set VXORACLE_MISTRAL__API_KEY, "
            "MISTRAL_API_KEY or LLM_API_KEY; see `voxoracle doctor`)"
        )

    from voxoracle.audio.endpoint import WebRtcVad
    from voxoracle.audio.sounddevice_backend import SoundDeviceBackend
    from voxoracle.stt.mistral import MistralTranscriber
    from voxoracle.tts.mistral import MistralSpeechSynthesizer
    from voxoracle.tts.player import SpeechPlayer
    from voxoracle.wakeword.detector import (
        OpenWakeWordDetector,
        OpenWakeWordScorer,
        resolve_model_path,
    )

    backend = SoundDeviceBackend(settings.audio)
    frame_samples = round(settings.audio.sample_rate * settings.audio.frame_ms / 1000)
    audio_input = backend.open_configured_input()
    audio_output = backend.open_configured_output(settings.tts.sample_rate)

    model_path = resolve_model_path(settings.wakeword.model, resolve_models_dir(settings))
    scorer = OpenWakeWordScorer(model_path)
    detector = OpenWakeWordDetector(
        scorer,
        model_name=scorer.model_name,
        threshold=settings.wakeword.threshold,
    )

    mistral_client = MistralAudioClient(
        api_key,
        base_url=settings.mistral.base_url,
        timeout=settings.mistral.timeout,
        retries=settings.mistral.retries,
    )
    transcriber: Transcriber = MistralTranscriber(
        mistral_client, model=settings.stt.model, language=settings.stt.language
    )
    synthesizer: Synthesizer = MistralSpeechSynthesizer(
        mistral_client,
        model=settings.tts.model,
        language=settings.tts.language,
        voice=settings.tts.voice,
        sample_rate=settings.tts.sample_rate,
    )
    player = SpeechPlayer(synthesizer, audio_output)

    docoracle = DocOracleClient(
        base_url=settings.docoracle.url,
        timeout=settings.docoracle.timeout,
    )

    session = VoiceSession(
        detector=detector,
        audio_input=audio_input,
        vad=WebRtcVad(settings.audio.vad_aggressiveness),
        transcriber=transcriber,
        client=docoracle,
        player=player,
        config=SessionConfig(
            sample_rate=settings.audio.sample_rate,
            frame_samples=frame_samples,
            max_record_seconds=settings.session.max_record_seconds,
            endpoint_silence_ms=settings.audio.endpoint_silence_ms,
            barge_in=settings.session.barge_in,
            follow_up=settings.session.follow_up,
            follow_up_seconds=settings.session.follow_up_seconds,
            language=settings.stt.language,
            voice=settings.tts.voice,
        ),
    )
    return SessionComponents(
        session=session,
        audio_input=audio_input,
        audio_output=audio_output,
        mistral_client=mistral_client,
        docoracle_client=docoracle,
    )
