"""Streaming playback of synthesized speech with barge-in cancellation.

:class:`SpeechPlayer` drives a :class:`~voxoracle.tts.protocol.Synthesizer` into
the WP2 :class:`~voxoracle.audio.protocols.AudioOutput`, writing each
:class:`~voxoracle.tts.protocol.SpeechChunk` as soon as it arrives so playback
begins before the provider finishes synthesizing the whole answer.

Barge-in: :meth:`SpeechPlayer.interrupt` is the stop hook. WP6 wires it to the
wake-word detector (and/or speech detection) so that hearing the activation word
during playback cuts the answer short. The player checks the interrupt between
chunks, stops the output device and abandons the provider stream, which cancels
the in-flight synthesis request.
"""

from __future__ import annotations

from contextlib import aclosing

from voxoracle.audio.protocols import AudioOutput
from voxoracle.tts.protocol import Synthesizer


class SpeechPlayer:
    """Streams a synthesizer's chunks to an output device, cancellably."""

    def __init__(self, synthesizer: Synthesizer, output: AudioOutput) -> None:
        self._synthesizer = synthesizer
        self._output = output
        self._interrupted = False

    @property
    def interrupted(self) -> bool:
        """Whether the last :meth:`speak` was stopped by :meth:`interrupt`."""
        return self._interrupted

    def interrupt(self) -> None:
        """Request barge-in: stop the current playback and the provider request.

        Safe to call at any time, including when nothing is playing; the flag is
        reset at the start of the next :meth:`speak`.
        """
        self._interrupted = True

    async def speak(self, text: str, *, voice: str | None = None) -> None:
        """Synthesize ``text`` and play it, stopping early if interrupted.

        Chunks are written to the output as they arrive. An :meth:`interrupt`
        between chunks stops the device and closes the provider stream, so no
        further audio is fetched. Returns once playback finishes or is stopped.
        """
        self._interrupted = False
        async with aclosing(self._synthesizer.synthesize(text, voice=voice)) as stream:
            async for chunk in stream:
                if self._interrupted:
                    break
                self._output.write(chunk.samples)
        if self._interrupted:
            self._output.stop()
