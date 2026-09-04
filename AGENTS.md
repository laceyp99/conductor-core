# Conductor Core Agent Guide

Treat these instructions as good defaults rather than hard rules. Explicit
developer instructions take precedence.

## Scope

This repository owns the reusable generation engine: public request/result
contracts, provider routing and adapters, model metadata, music models, MIDI
conversion, artifact storage, and optional playback helpers. Do not introduce
Gradio, Dash, or evaluation dependencies into Core.

Core converts provider-specific responses into provider-independent music
models and artifacts. Keep provider differences at the routing and adapter
boundary so the engine, storage, and consumers operate on stable shared
contracts.

## Consumers and severity

Core is consumed by other pre-release Conductor applications, scripts, and
notebooks with a small user base. Prefer sound architecture and correct logic
over preserving every early API shape. When practical, deprecate behavior for
one release before removing it in the next.

Import failures and corrupted or deleted generation history are severe.
Optional MP3 rendering failure is intentionally non-fatal when MIDI generation
succeeds.

## Glossary

| Term | Definition |
| --- | --- |
| **Core** | The reusable Python package, not a Conductor UI. |
| **Consumer** | An application, script, notebook, or service that imports Core. |
| **Provider** | An OpenAI, Anthropic, Google, or Ollama integration. |
| **Loop** | Core's validated, provider-independent four-bar music representation. |
| **Generation workspace** | The persisted files belonging to one generation. |
| **Audio preview** | Optional MP3 output; MIDI remains the primary generated artifact. |

## Key paths

- `src/conductor_core/engine.py`: end-to-end synchronous generation.
- `src/conductor_core/config.py`: public configuration and result contracts.
- `src/conductor_core/resources/model_list.json`: canonical provider metadata.
- `src/conductor_core/providers/`: provider request and parsing behavior.
- `src/conductor_core/midi.py`: loop/MIDI conversion.
- `src/conductor_core/storage.py`: filesystem artifacts and history.
- `tests/`: deterministic unit and package-boundary tests.
- `.agents/skills/add-model-support/`: model-support workflow.

## Working rules

Follow the setup and validation commands in
[`docs/development.md`](docs/development.md). While iterating, run the narrowest
relevant test module first. Before handing off a change, run the full documented
checks unless the task or environment prevents it; report anything skipped.

- Inspect existing provider and test patterns before editing.
- Keep provider services lazy and import-safe; never require a live service at import time.
- Prefer metadata-driven model capabilities over hard-coded model exceptions.
- Do not make live provider calls or run broad evaluations unless explicitly requested.
- Treat FluidSynth and FFmpeg as optional external tools.
- Do not commit generated MIDI, audio, histories, build output, or credentials.
- Keep relevant documentation and the changelog in sync with behavior changes.

## Pull requests

- Never open a pull request unless the developer explicitly asks.
- Use a plain-language Conventional Commit title, such as
  `fix(validation): reject invalid loop timing`.
- State the problem in one or two sentences, then explain the fix.
- Rebase onto the latest `main` before opening a pull request.
- List validation performed and anything skipped.
- Do not include generated artifacts, credentials, or unrelated cleanup.
