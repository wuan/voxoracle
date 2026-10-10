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
during playback cuts the answer short. :meth:`interrupt` sets an event that
:meth:`speak` races against the next chunk, so playback stops promptly and the
in-flight provider request is cancelled even if the provider goes quiet. Call
:meth:`interrupt` from the event-loop thread (the session loop does).

The device write blocks until PortAudio has played the samples, so it runs in a
worker thread (``asyncio.to_thread``) to keep the event loop free; otherwise the
barge-in detector could not run while a chunk plays.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing

from voxoracle.audio.protocols import AudioOutput
from voxoracle.audio.resample import resample
from voxoracle.tts.protocol import SpeechChunk, Synthesizer


class SpeechPlayer:
    """Streams a synthesizer's chunks to an output device, cancellably."""

    def __init__(self, synthesizer: Synthesizer, output: AudioOutput) -> None:
        self._synthesizer = synthesizer
        self._output = output
        self._interrupted = False
        self._stop: asyncio.Event | None = None

    @property
    def interrupted(self) -> bool:
        """Whether the last :meth:`speak` was stopped by :meth:`interrupt`."""
        return self._interrupted

    def interrupt(self) -> None:
        """Request barge-in: stop playback now and cancel the provider request.

        Sets the flag and, if a :meth:`speak` is in progress, the stop event it
        races against, so the pending chunk fetch is abandoned without waiting for
        the provider or the read timeout. Safe to call when nothing is playing;
        the flag is reset at the start of the next :meth:`speak`.
        """
        self._interrupted = True
        if self._stop is not None:
            self._stop.set()

    async def speak(self, text: str, *, voice: str | None = None) -> None:
        """Synthesize ``text`` and play it, stopping promptly if interrupted.

        Each chunk is resampled to the output's source rate and written in a
        worker thread (the device write blocks for the chunk's playback
        duration). While waiting for the next chunk the player also waits on the
        interrupt event, so barge-in stops playback and closes the provider
        stream at once, cancelling the in-flight synthesis request. Returns once
        playback finishes or is stopped.
        """
        self._interrupted = False
        self._stop = asyncio.Event()
        target_rate = self._output.sample_rate
        async with aclosing(self._synthesizer.synthesize(text, voice=voice)) as stream:
            while not self._interrupted:
                chunk = await self._next_chunk(stream)
                if chunk is None:
                    break
                samples = resample(chunk.samples, chunk.sample_rate, target_rate)
                await asyncio.to_thread(self._output.write, samples)
        if self._interrupted:
            self._output.stop()
        self._stop = None

    async def _next_chunk(self, stream: AsyncGenerator[SpeechChunk]) -> SpeechChunk | None:
        """Return the next chunk, or ``None`` at end of stream or on interrupt.

        Races the chunk fetch against the interrupt event: if barge-in is
        requested while the provider is quiet, the pending fetch is cancelled
        (which closes the response) instead of blocking on the read timeout.
        """
        assert self._stop is not None
        next_chunk = asyncio.ensure_future(anext(stream))
        stop = asyncio.ensure_future(self._stop.wait())
        done, _ = await asyncio.wait({next_chunk, stop}, return_when=asyncio.FIRST_COMPLETED)
        if stop in done and self._interrupted:
            next_chunk.cancel()
            await asyncio.gather(next_chunk, return_exceptions=True)
            return None
        stop.cancel()
        await asyncio.gather(stop, return_exceptions=True)
        try:
            return next_chunk.result()
        except StopAsyncIteration:
            return None
