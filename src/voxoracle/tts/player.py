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
during playback cuts the answer short. ``interrupt`` sets a flag and an event
and stops the output at once; ``speak`` races the next chunk fetch against the
event (so a quiet provider does not delay cancellation) and writes audio in
short slices checked against the flag (so a long chunk - e.g. a whole WAV answer
- stops within a bounded fraction of a second rather than after the chunk plays
out). Call ``interrupt`` from the event-loop thread (the session loop does).

The device write blocks until PortAudio has played the samples, so it runs in a
worker thread (``asyncio.to_thread``) to keep the event loop free; otherwise the
barge-in detector could not run while a chunk plays. Slicing keeps each blocking
write short, so the flag is re-checked promptly, and barge-in aborts the output
(``Pa_AbortStream``) so already-buffered audio is discarded rather than drained.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing

import numpy as np
from numpy.typing import NDArray

from voxoracle.audio.protocols import AudioOutput
from voxoracle.audio.resample import resample
from voxoracle.tts.protocol import SpeechChunk, Synthesizer

_LOGGER = logging.getLogger(__name__)

#: Longest audio slice handed to a single blocking device write. Bounds how long
#: barge-in can wait for an in-flight write to return, independent of chunk size.
PLAY_SLICE_SECONDS = 0.1


class SpeechPlayer:
    """Streams a synthesizer's chunks to an output device, cancellably."""

    def __init__(self, synthesizer: Synthesizer, output: AudioOutput) -> None:
        self._synthesizer = synthesizer
        self._output = output
        self._interrupted = False
        self._speaking = False
        self._stop: asyncio.Event | None = None
        # True once the output has been aborted, so the next speak() restarts it.
        self._output_aborted = False

    @property
    def interrupted(self) -> bool:
        """Whether the last :meth:`speak` was stopped by :meth:`interrupt`."""
        return self._interrupted

    def interrupt(self) -> None:
        """Request barge-in: stop playback now and cancel the provider request.

        Sets the flag and, if a :meth:`speak` is in progress, the stop event it
        races against, and aborts the output device immediately so audio that is
        already playing (including a blocking device write and any audio already
        buffered) ceases at once. The next :meth:`speak` restarts the output.
        Safe to call when nothing is playing; the flag is reset at the start of
        the next :meth:`speak`.
        """
        self._interrupted = True
        if self._stop is not None:
            self._stop.set()
        if self._speaking:
            self._abort_output()

    def _abort_output(self) -> None:
        """Abort the output, discarding buffered audio.

        Prefers the protocol's :meth:`~voxoracle.audio.protocols.AudioOutput.abort`
        (``Pa_AbortStream``: discards the buffer) and falls back to ``stop`` for
        simpler sinks that only drain. Records that the output must be restarted
        before it can play again.
        """
        abort = getattr(self._output, "abort", None)
        if callable(abort):
            abort()
        else:  # pragma: no cover - every real output implements abort
            self._output.stop()
        self._output_aborted = True

    def _resume_output(self) -> None:
        """Restart the output after an abort, so the next utterance plays."""
        if not self._output_aborted:
            return
        start = getattr(self._output, "start", None)
        if callable(start):
            start()
        self._output_aborted = False

    async def speak(self, text: str, *, voice: str | None = None) -> None:
        """Synthesize ``text`` and play it, stopping promptly if interrupted.

        Each chunk is resampled to the output's source rate and written in short
        slices in a worker thread (the device write blocks for the slice's
        playback duration), checking the interrupt flag between slices. While
        waiting for the next chunk the player also waits on the interrupt event,
        so barge-in stops playback and closes the provider stream at once,
        cancelling the in-flight synthesis request. A previous barge-in left the
        output aborted, so it is restarted first so this utterance plays.
        Returns once playback finishes or is stopped.
        """
        self._interrupted = False
        self._speaking = True
        self._stop = asyncio.Event()
        self._resume_output()
        target_rate = self._output.sample_rate
        try:
            async with aclosing(self._synthesizer.synthesize(text, voice=voice)) as stream:
                while not self._interrupted:
                    chunk = await self._next_chunk(stream)
                    if chunk is None:
                        break
                    samples = resample(chunk.samples, chunk.sample_rate, target_rate)
                    await self._play(samples, target_rate)
        finally:
            self._speaking = False
            # Always drop the stop event so interrupt() cannot touch a stale one.
            self._stop = None

    async def _play(self, samples: NDArray[np.int16], sample_rate: int) -> None:
        """Write ``samples`` in bounded slices, stopping on barge-in.

        A slice write may fail because we aborted the stream (``interrupt``);
        such a failure is swallowed once the interrupt is set, since the output
        was deliberately aborted. Any other write failure propagates.
        """
        slice_samples = max(1, round(sample_rate * PLAY_SLICE_SECONDS))
        for start in range(0, samples.size, slice_samples):
            if self._interrupted:
                self._abort_output()
                return
            try:
                await asyncio.to_thread(self._output.write, samples[start : start + slice_samples])
            except Exception:
                if not self._interrupted:
                    raise
                # Our own abort made the blocking write fail; that is expected.
                _LOGGER.debug("device write failed after barge-in abort", exc_info=True)
                return

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
            # Barge-in wins: abandon the in-flight fetch. If the fetch had already
            # produced a chunk (or raised), it is intentionally discarded here -
            # the user interrupted, so a late chunk or its error is not surfaced.
            next_chunk.cancel()
            await asyncio.gather(next_chunk, return_exceptions=True)
            return None
        stop.cancel()
        await asyncio.gather(stop, return_exceptions=True)
        try:
            return next_chunk.result()
        except StopAsyncIteration:
            return None
