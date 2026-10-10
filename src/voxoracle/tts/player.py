"""Streaming playback of synthesized speech with barge-in cancellation.

:class:`SpeechPlayer` drives a :class:`~voxoracle.tts.protocol.Synthesizer` into
the WP2 :class:`~voxoracle.audio.protocols.AudioOutput`, writing each
:class:`~voxoracle.tts.protocol.SpeechChunk` as soon as it arrives so playback
begins before the provider finishes synthesizing the whole answer.

Chunks carry their own sample rate (the provider's), which may differ from the
rate the output was opened for; the player resamples each chunk to the output
rate before writing, so a 24 kHz stream plays at the right speed and pitch on a
16 kHz output.

Barge-in: :meth:`SpeechPlayer.interrupt` is the stop hook. WP6 wires it to the
wake-word detector (and/or speech detection) so that hearing the activation word
during playback cuts the answer short. The player checks the interrupt between
chunks, stops the output device and abandons the provider stream, which cancels
the in-flight synthesis request.

The device write blocks until PortAudio has played the samples, so it runs in a
worker thread (``asyncio.to_thread``) to keep the event loop free; otherwise the
barge-in detector could not run while a chunk plays.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing

from voxoracle.audio.protocols import AudioOutput
from voxoracle.audio.resample import resample
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

        Each chunk is resampled to the output's source rate before it is written,
        so the provider's rate never has to match the device. Chunks are written
        as they arrive, each in a worker thread because the device write blocks
        for the chunk's playback duration. An :meth:`interrupt` between chunks
        stops the device and closes the provider stream, so no further audio is
        fetched. Returns once playback finishes or is stopped.
        """
        self._interrupted = False
        target_rate = self._output.sample_rate
        async with aclosing(self._synthesizer.synthesize(text, voice=voice)) as stream:
            async for chunk in stream:
                if self._interrupted:
                    break
                samples = resample(chunk.samples, chunk.sample_rate, target_rate)
                await asyncio.to_thread(self._output.write, samples)
        if self._interrupted:
            self._output.stop()
