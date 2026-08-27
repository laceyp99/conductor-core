# Conductor command-line interface

The `conductor` command is the supported command-line interface for Conductor
Core. It serves interactive use and automation while keeping generation,
provider routing, validation, and artifact storage in the Core engine. The same
application runs as `python -m conductor_core`.

## Installation and credentials

Install the base package plus the provider extras you need. For example:

```powershell
python -m pip install "conductor-core[openai]"
```

Hosted generation reads credentials from Core's existing environment fallback:

| Provider | Environment variable |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GEMINI_API_KEY` |
| Ollama | `OLLAMA_API_HOST_ADDRESS` (optional host override) |

Credentials are deliberately not accepted as command options, which keeps them
out of shell history and process listings.

## Discover models offline

`models` reads the model registry packaged in the installed wheel. It does not
initialize a provider client, contact a service, or require credentials.

```powershell
conductor models
conductor models --provider google
conductor models --provider OPENAI --json
```

Provider matching is case-insensitive. An unknown provider is a usage error.

| Option | Behavior |
|---|---|
| `--provider TEXT` | Return only one provider's packaged models |
| `--json` | Write a schema-version-1 model envelope to stdout |
| `--quiet` | Suppress human model output; JSON is never suppressed |

Each JSON model record contains `provider`, `model`, `extended_thinking`,
`always_on_adaptive_thinking`, `effort_options`, `temperature_supported`,
`max_tokens`, `max_thinking_budget`, `cost`, and `rate_limits`. Optional or
unsupported capabilities use `null`, `false`, or an empty list rather than
being omitted.

```json
{
  "schema_version": 1,
  "status": "success",
  "models": [
    {
      "provider": "OpenAI",
      "model": "gpt-4o-mini",
      "extended_thinking": false,
      "always_on_adaptive_thinking": false,
      "effort_options": [],
      "temperature_supported": true,
      "max_tokens": 16384,
      "max_thinking_budget": null,
      "cost": {"input": 0.15, "cached input": 0.075, "output": 0.6},
      "rate_limits": {"RPM": 500, "TPM": 30000, "RPD": null}
    }
  ]
}
```

## Generate one loop

`generate` performs exactly one request and persists the complete managed
generation workspace. It may contact a paid provider and incur usage charges.

```powershell
conductor generate --key C --scale Major `
  --description "warm neo-soul electric piano chords" `
  --model gpt-4o-mini --temperature 0.3
```

The musical inputs are named and required:

| Option | Behavior |
|---|---|
| `--key TEXT` | Musical key accepted by `GenerationRequest` |
| `--scale TEXT` | Scale name accepted case-insensitively by Core |
| `--description TEXT` | User description used to request the loop |
| `--model TEXT` | Hosted or available Ollama model name |

Optional request controls map directly to `GenerationRequest`:

| Option | Default | Behavior |
|---|---:|---|
| `--temperature FLOAT` | `0.0` | Sampling temperature from 0.0 through 2.0 |
| `--use-thinking` | off | Request extended reasoning |
| `--effort TEXT` | unset | Provider reasoning-effort value |
| `--prompt TEXT` | unset | Inline system-prompt override |
| `--prompt-file FILE` | unset | Strict UTF-8 system-prompt override file |
| `--render-audio` | off | Attempt MP3 rendering after MIDI generation |
| `--soundfont-path FILE` | Core default | SoundFont for requested audio rendering |

`--prompt` and `--prompt-file` are mutually exclusive. Prompt input from stdin
is not supported. The CLI does not provide batch generation or an exact output
filename; artifacts remain together in Core's managed generation history.

Engine and output controls are:

| Option | Default | Behavior |
|---|---:|---|
| `--artifact-root DIRECTORY` | Core data directory | Select the generation-history root |
| `--request-timeout FLOAT` | provider default | Set a positive timeout in seconds |
| `--max-generations INTEGER` | `20` | Retain this many recent generations |
| `--json` | off | Write one schema-version-1 result to stdout |
| `--quiet` | off | Suppress human success and progress output |
| `--verbose` | off | Show progress when stderr is redirected |

The root `--debug` option must appear before the command, such as
`conductor --debug generate ...`. It adds a traceback for unexpected internal
failures. It does not add progress; use `--verbose` for that.

## Human output, streams, and progress

Human success output is plain text on stdout and reports the generation ID,
absolute MIDI, audio, and message paths, provider cost (or `unavailable`), and
warnings. Errors and progress are written to stderr.

Progress appears only when all of these are true:

- human output is selected;
- quiet mode is off; and
- stderr is a terminal, or `--verbose` was supplied.

JSON mode never emits progress, even with `--verbose`, so stdout remains one
parseable result. `--quiet` never suppresses JSON results or any errors.

Audio rendering is optional. If MIDI generation succeeds but requested audio
is skipped or fails, the command still exits successfully, reports no audio
path, and includes Core's warning in human or JSON output.

## Generation JSON schema version 1

`generate --json` writes one compact JSON object to stdout. It includes the
confirmed non-secret request fields, selected generation metadata, absolute
artifact paths, a provider-independent loop, cost, and warnings.

```json
{
  "schema_version": 1,
  "status": "success",
  "request": {
    "key": "C",
    "scale": "Major",
    "description": "warm neo-soul chords",
    "model": "gpt-4o-mini",
    "temperature": 0.3,
    "use_thinking": false,
    "effort": null,
    "render_audio": false
  },
  "metadata": {
    "generation_id": "20260827_120000_abcd1234",
    "provider": "OpenAI",
    "timestamp": "2026-08-27T12:00:00+00:00",
    "soundfont": null,
    "audio_render_succeeded": false
  },
  "artifacts": {
    "midi": "C:\\data\\gen_abc\\loop.mid",
    "audio": null,
    "messages": "C:\\data\\gen_abc\\messages.json"
  },
  "loop": {
    "Bar_1": {"num": 1, "notes": []},
    "Bar_2": {"num": 2, "notes": []},
    "Bar_3": {"num": 3, "notes": []},
    "Bar_4": {"num": 4, "notes": []}
  },
  "cost": 0.00012,
  "warnings": []
}
```

Every note in `loop` has `pitch`, `octave`, `velocity`, and a `time` object.
Both `time.start_beat` and `time.duration` are integers, including for Google
models whose internal response schema uses string timing enums. Raw provider
messages, prompt-override text, and the input SoundFont path are not included.

Schema version 1 allows additive fields. Removing or renaming a field, changing
its JSON type, or changing its meaning requires a new schema version.

## Errors and exit codes

| Exit | Category | Examples |
|---:|---|---|
| `0` | Success | MIDI succeeded, including non-fatal audio warnings |
| `1` | Unexpected | Internal defect not covered by a known boundary |
| `2` | Usage or validation | Missing options, invalid request, unknown model/provider |
| `3` | Configuration or dependency | Missing/rejected credential or provider SDK |
| `4` | Provider or generation | Rate limit, timeout, connection, or rejected request |

Normal failures are concise and go to stderr. With `--json`, failures use this
schema-versioned stderr envelope and leave stdout empty:

```json
{
  "schema_version": 1,
  "status": "error",
  "error": {
    "code": "configuration_error",
    "message": "OpenAI failed: an API key is required"
  }
}
```

Stable machine error codes include `usage_error`, `validation_error`,
`configuration_error`, `dependency_error`, `generation_error`, and
`unexpected_error`.

## Troubleshooting

- Run `conductor --version` and `python -m conductor_core --version`; they
  should report the same installed distribution version.
- Use `conductor models` to confirm a hosted model name without contacting its
  provider. Ollama models are discovered only during generation routing.
- Exit 3 with a dependency error means the matching provider extra is missing.
- Audio warnings do not mean MIDI failed. Check the MIDI path, then install the
  `playback` extra and ensure FluidSynth and FFmpeg are on `PATH`.
- Use `--verbose` for progress in redirected logs. Use root `--debug` only when
  diagnosing an unexpected exit 1.
