# Conductor Core Agent Guide

Think of these instructions less as "hard rules", and more as "good defaults". The developer's preferences should be able to override anything here.

## Scope

This repository owns the reusable generation engine: public request/result
contracts, provider routing and adapters, model metadata, music models, MIDI
conversion, artifact storage, and optional playback helpers. Do not introduce
Gradio, Dash, or evaluation dependencies into Core.
Core converts provider-specific responses into provider-independent music models and artifacts. Keep provider differences at the routing/adapter boundary so the engine, storage, and consumers operate on stable shared contracts.

## Consumer

Core is consumed by other Conductor applications, scripts, and notebooks. 
These repositories are all within the pre-release stage with a small userbase.
Focus more on getting the right architecture or logic, over ensuring backwards compatability.
Favor using deprecation warnings within a version release and then quickly remove within the next release.
Optional MP3 rendering failing while MIDI succeeds is intentionally non-fatal.
Import failures and corrupted/deleted history is severe

## Glossary
| Term | Definition |
| ------------ | --------------------------------------------------------------------- |
| **Core** | The reusable Python package, not a Conductor UI. |
| **Consumer** | An application, script, notebook, or service that imports Core. |
| **Provider** | An OpenAI, Anthropic, Google, or Ollama integration. |
| **Loop** | Core’s validated, provider-independent four-bar music representation. |
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

Follow the development setup and full validation commands in the [README.md](README.md#contribute-to-core). While iterating, run the narrowest relevant test module first. Before handing off a change, run the full documented checks unless the task or environment prevents it; report anything skipped.

- Inspect existing provider and test patterns before editing.
- Keep provider services lazy and import-safe; never require a live service at import time.
- Prefer metadata-driven model capabilities over hard-coded model exceptions.
- Do not make live provider calls or run broad evaluations unless explicitly requested.
- Treat FluidSynth and FFmpeg as optional external tools.
- Do not commit generated MIDI, audio, histories, build output, or credentials.
- Make sure to update docs like README.md, pyproject.toml, and CHANGELOG.md before pushing.

## PR Ettiquite 

- Never make a PR unless the developer explicitly asks you to do so.
- Conventional commit titles, plain language: `fix(validation): reject invalid loop timing`
- Body: the problem in a sentence or two, then how you fixed it.
- Rebase onto latest main before opening. Stale branches conflict and burn a review round.
- List validation performed and anything skipped.
- Do not include generated artifacts, credentials, or unrelated cleanup.