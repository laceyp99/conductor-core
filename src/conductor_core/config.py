"""Public configuration and request/result contracts for Conductor Core."""

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

from conductor_core.music import ENHARMONIC_NOTE_NAMES, SCALE_INTERVALS
from conductor_core.paths import resolve_default_artifact_root
from conductor_core.storage import MAX_GENERATIONS, _validate_max_generations


@dataclass(frozen=True)
class ProviderCredentials:
    """Provider credentials supplied by an app or script."""

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    ollama_host: str | None = None


@dataclass(frozen=True)
class EngineConfig:
    """Environment, resource, and storage configuration for the core engine."""

    artifact_root: str | Path = field(default_factory=resolve_default_artifact_root)
    provider_credentials: ProviderCredentials = field(
        default_factory=ProviderCredentials
    )
    prompt_override: str | None = None
    default_soundfont_path: str | Path | None = None
    max_generations: int | None = MAX_GENERATIONS
    request_timeout: float | None = None

    def __post_init__(self) -> None:
        _validate_max_generations(self.max_generations)
        if self.request_timeout is not None and (
            isinstance(self.request_timeout, bool)
            or not isinstance(self.request_timeout, (int, float))
            or not isfinite(self.request_timeout)
            or self.request_timeout <= 0
        ):
            raise ValueError("request_timeout must be None or a positive finite number")

    @classmethod
    def from_defaults(
        cls,
        artifact_root: str | Path | None = None,
        provider_credentials: ProviderCredentials | None = None,
        prompt_override: str | None = None,
        default_soundfont_path: str | Path | None = None,
        max_generations: int | None = MAX_GENERATIONS,
        request_timeout: float | None = None,
    ) -> "EngineConfig":
        """Create a config using Core defaults plus caller-provided overrides."""
        return cls(
            artifact_root=(
                artifact_root
                if artifact_root is not None
                else resolve_default_artifact_root()
            ),
            provider_credentials=provider_credentials or ProviderCredentials(),
            prompt_override=prompt_override,
            default_soundfont_path=default_soundfont_path,
            max_generations=max_generations,
            request_timeout=request_timeout,
        )


@dataclass(frozen=True)
class GenerationRequest:
    """One prompt-to-loop generation request.

    For models with configurable reasoning effort, ``use_thinking=False``
    selects the model's lowest supported effort. The requested ``effort`` is
    used unchanged when thinking is enabled.
    """

    key: str
    scale: str
    description: str
    model: str
    temperature: float = 0.0
    use_thinking: bool = False
    effort: str = "low"
    prompt_override: str | None = None
    render_audio: bool = False
    soundfont_path: str | Path | None = None

    def __post_init__(self) -> None:
        valid_keys = tuple(note for notes in ENHARMONIC_NOTE_NAMES for note in notes)
        if self.key not in valid_keys:
            raise ValueError(
                f"Invalid key {self.key!r}. Expected one of: {', '.join(valid_keys)}"
            )

        if not isinstance(self.scale, str) or self.scale.lower() not in SCALE_INTERVALS:
            raise ValueError(
                f"Invalid scale {self.scale!r}. Expected one of: "
                f"{', '.join(SCALE_INTERVALS)} (case-insensitive)"
            )

        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not isfinite(self.temperature)
            or not 0.0 <= self.temperature <= 2.0
        ):
            raise ValueError(
                f"Invalid temperature {self.temperature!r}. "
                "Expected a finite number between 0.0 and 2.0 (inclusive)"
            )


@dataclass(frozen=True)
class ProgressEvent:
    """Structured progress event emitted by the synchronous engine."""

    stage: str
    message: str
    detail: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    """Complete result of a generated loop and its persisted artifacts."""

    generation_id: str
    loop: Any
    midi_path: str
    audio_path: str | None
    messages: list[dict[str, Any]]
    cost: float | None
    metadata: Any
    warnings: list[str] = field(default_factory=list)
