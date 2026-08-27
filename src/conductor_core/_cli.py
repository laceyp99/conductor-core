"""Pure, private adapters for Conductor's command-line interface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from conductor_core.errors import (
    ProviderAuthenticationError,
    ProviderError,
)
from conductor_core.models import (
    DURATION_SIXTEENTH_G_TO_INT,
    SIXTEENTH_NOTE_G_TO_INT,
)

SCHEMA_VERSION = 1

EXIT_UNEXPECTED = 1
EXIT_USAGE = 2
EXIT_CONFIGURATION = 3
EXIT_GENERATION = 4


@dataclass(frozen=True)
class CliError:
    """A stable CLI error classification."""

    exit_code: int
    code: str
    message: str


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _numeric_timing(value: Any, mapping: dict[str, int]) -> int:
    raw_value = _enum_value(value)
    return mapping.get(raw_value, raw_value)


def normalize_loop(loop: Any) -> dict[str, Any]:
    """Return the provider-independent, numeric four-bar loop representation."""
    normalized = {}
    for bar_number in range(1, 5):
        field_name = f"Bar_{bar_number}"
        bar = getattr(loop, field_name)
        normalized[field_name] = {
            "num": bar.num,
            "notes": [
                {
                    "pitch": note.pitch,
                    "octave": note.octave,
                    "velocity": note.velocity,
                    "time": {
                        "start_beat": _numeric_timing(
                            note.time.start_beat, SIXTEENTH_NOTE_G_TO_INT
                        ),
                        "duration": _numeric_timing(
                            note.time.duration, DURATION_SIXTEENTH_G_TO_INT
                        ),
                    },
                }
                for note in bar.notes
            ],
        }
    return normalized


def normalize_model_records(
    model_info: dict[str, Any], provider: str | None = None
) -> list[dict[str, Any]]:
    """Flatten packaged model metadata into deterministic public CLI records."""
    models_by_provider = model_info["models"]
    selected_provider = None
    if provider is not None:
        selected_provider = next(
            (
                name
                for name in models_by_provider
                if name.casefold() == provider.casefold()
            ),
            None,
        )
        if selected_provider is None:
            choices = ", ".join(sorted(models_by_provider))
            raise ValueError(
                f"Unknown provider {provider!r}. Expected one of: {choices}"
            )

    records = []
    for provider_name in sorted(models_by_provider):
        if selected_provider is not None and provider_name != selected_provider:
            continue
        for model_name in sorted(models_by_provider[provider_name]):
            config = models_by_provider[provider_name][model_name]
            records.append(
                {
                    "provider": provider_name,
                    "model": model_name,
                    "extended_thinking": bool(config.get("extended_thinking", False)),
                    "always_on_adaptive_thinking": bool(
                        config.get("always_on_adaptive_thinking", False)
                    ),
                    "effort_options": list(config.get("effort_options", [])),
                    "temperature_supported": config.get("temperature_supported", True),
                    "max_tokens": config.get("max_tokens"),
                    "max_thinking_budget": config.get("max_thinking_budget"),
                    "cost": config.get("cost"),
                    "rate_limits": config.get("rate_limits"),
                }
            )
    return records


def absolute_path(path: str | None) -> str | None:
    """Resolve a reported artifact path without requiring it to exist."""
    if path is None:
        return None
    return str(Path(path).expanduser().resolve())


def success_envelope(request: Any, result: Any) -> dict[str, Any]:
    """Build schema-version-1 generation output."""
    metadata = result.metadata
    timestamp = metadata.timestamp
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "request": {
            "key": request.key,
            "scale": request.scale,
            "description": request.description,
            "model": request.model,
            "temperature": request.temperature,
            "use_thinking": request.use_thinking,
            "effort": request.effort,
            "render_audio": request.render_audio,
        },
        "metadata": {
            "generation_id": result.generation_id,
            "provider": metadata.provider,
            "timestamp": timestamp.isoformat()
            if hasattr(timestamp, "isoformat")
            else str(timestamp),
            "soundfont": metadata.soundfont,
            "audio_render_succeeded": metadata.audio_path is not None,
        },
        "artifacts": {
            "midi": absolute_path(result.midi_path),
            "audio": absolute_path(result.audio_path),
            "messages": absolute_path(metadata.messages_path),
        },
        "loop": normalize_loop(result.loop),
        "cost": result.cost,
        "warnings": list(result.warnings),
    }


def models_envelope(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "models": records,
    }


def error_envelope(error: CliError) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "error": {"code": error.code, "message": error.message},
    }


def classify_error(exc: BaseException) -> CliError:
    """Map only known Core failure boundaries to stable CLI errors."""
    message = str(exc) or type(exc).__name__
    if isinstance(exc, ProviderAuthenticationError):
        return CliError(EXIT_CONFIGURATION, "configuration_error", message)
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return CliError(EXIT_CONFIGURATION, "dependency_error", message)
    if isinstance(exc, ProviderError):
        return CliError(EXIT_GENERATION, "generation_error", message)
    if isinstance(exc, ValueError) and (
        message == "Invalid Model Selected"
        or message
        == "Invalid Model Selected. If you intended to use Ollama, it is currently unavailable."
        or (message.startswith("Invalid effort ") and ". Expected one of: " in message)
    ):
        return CliError(EXIT_USAGE, "validation_error", message)
    return CliError(EXIT_UNEXPECTED, "unexpected_error", message)


def should_show_progress(
    *, json_output: bool, quiet: bool, verbose: bool, stderr_isatty: bool
) -> bool:
    """Apply the CLI's stdout/stderr-safe progress policy."""
    return not json_output and not quiet and (verbose or stderr_isatty)


def format_generation(result: Any) -> str:
    """Format a successful generation as portable plain text."""
    cost = "unavailable" if result.cost is None else str(result.cost)
    lines = [
        f"Generation: {result.generation_id}",
        f"MIDI: {absolute_path(result.midi_path)}",
        f"Audio: {absolute_path(result.audio_path) or 'not available'}",
        f"Messages: {absolute_path(result.metadata.messages_path)}",
        f"Cost: {cost}",
    ]
    lines.extend(f"Warning: {warning}" for warning in result.warnings)
    return "\n".join(lines)


def format_models(records: list[dict[str, Any]]) -> str:
    """Format offline model discovery as portable plain text."""
    lines = []
    current_provider = None
    for record in records:
        if record["provider"] != current_provider:
            current_provider = record["provider"]
            if lines:
                lines.append("")
            lines.append(f"{current_provider}:")
        capabilities = []
        if record["extended_thinking"]:
            capabilities.append("thinking")
        if record["effort_options"]:
            capabilities.append("effort=" + ",".join(record["effort_options"]))
        suffix = f" ({'; '.join(capabilities)})" if capabilities else ""
        lines.append(f"  {record['model']}{suffix}")
    return "\n".join(lines)
