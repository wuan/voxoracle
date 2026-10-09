# VoxOracle

The voice frontend for DocOracle (https://github.com/wuan/docoracle) — ask your documentation out loud.

VoxOracle puts a spoken conversational interface on top of DocOracle, an experimental RAG/chatbot solution that provides natural-language access to static Antora/AsciiDoc documentation. Speak your question; VoxOracle transcribes it, sends it to DocOracle's /ask API, and reads the answer back — with sources available on screen or on request.

Named in the spirit of its parent: if DocOracle is the oracle of your docs, VoxOracle is its voice.

How It Works

```
You (speech) ─▶ Speech-to-Text ─▶ DocOracle /ask ─▶ LLM + RAG retrieval
                                                            │
You (hearing) ◀─ Text-to-Speech ◀─ Answer + citations ◀─────┘
```

1. Capture — the browser captures your microphone input (Web Speech API or a pluggable STT backend).
2. Ask — the transcribed question is sent to a running DocOracle server via its HTTP API (POST /ask).
3. Retrieve & answer — DocOracle performs hybrid retrieval (semantic FAISS + German-aware BM25, fused via RRF) and generates a grounded, cited answer.
4. Speak — the answer is read back via the Web Speech API or a pluggable TTS backend, with barge-in support to interrupt.

Features

• Voice-first interaction — push-to-talk or continuous listening modes
• Grounded answers — every spoken answer carries DocOracle's citations (module:pages:page#section), displayed alongside the conversation
• Filtering by voice or UI — restrict questions to a module, component, or version
• Conversation history — local, per-device history of questions and answers
• Backend-agnostic — works with either DocOracle answer backend (engine or agent)
• Pluggable STT/TTS — browser-native speech by default, with hooks for external providers
• Fallback text mode — type instead of talk whenever speech isn't practical

Prerequisites

• A running DocOracle server (Python 3.12+, ingested documentation):
  
  docoracle serve --host 0.0.0.0 --port 8000

• A modern browser with microphone access (Chrome/Edge for full Web Speech API support)

• Optional: an external STT/TTS provider API key, if you configure one

Installation

pip install -e .
voxoracle serve              # start the frontend
voxoracle --docoracle-url http://localhost:8000

Note: Installation and CLI commands are the intended interface; adjust this section once the project scaffolding is in place.

Configuration

VoxOracle reads its settings from config.yaml:

docoracle:
  url: http://localhost:8000   # DocOracle server
  timeout: 120                # Seconds to wait for an answer

speech:
  stt: web-speech             # web-speech | external provider
  tts: web-speech             # web-speech | external provider
  language: de-DE             # Recognition locale
  barge_in: true              # Interrupt playback when you speak

session:
  history: local              # Conversation history storage

The recognition language defaults to German to match DocOracle's German-aware BM25 pipeline, but any locale supported by the speech backend can be configured.

Architecture

voxoracle/
├── src/voxoracle/
│   ├── client/          # DocOracle API client (ask, search, info)
│   ├── speech/          # STT/TTS abstraction layer + providers
│   ├── conversation/    # Session and history management
│   ├── server/          # Serves the web UI and proxies API calls
│   └── cli.py           # CLI interface
├── static/              # Voice-enabled web UI
└── tests/

VoxOracle deliberately contains no retrieval or LLM logic of its own — it is a pure presentation layer. All answering, retrieval, citation, and filtering behavior comes from DocOracle, which guarantees the two projects can evolve independently.

Development

pip install -e ".[dev]"
pre-commit install         # ruff, basedpyright, pytest on commit
pytest

Project Status

Experimental — just like DocOracle. Expect breaking changes; the API surface is not stable yet.

License

MIT

Acknowledgements

• DocOracle (https://github.com/wuan/docoracle) by wuan (https://github.com/wuan) — the RAG engine and documentation backend this project gives a voice to
