"""Public configuration and request/result contracts for Conductor Core."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conductor_core.paths import resolve_default_artifact_root
from conductor_core.storage import MAX_GENERATIONS


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
    provider_credentials: ProviderCredentials = field(default_factory=ProviderCredentials)
    prompt_override: str | None = None
    default_soundfont_path: str | Path | None = None
    max_generations: int | None = MAX_GENERATIONS

    @classmethod
    def from_defaults(
        cls,
        artifact_root: str | Path | None = None,
        provider_credentials: ProviderCredentials | None = None,
        prompt_override: str | None = None,
        default_soundfont_path: str | Path | None = None,
        max_generations: int | None = MAX_GENERATIONS,
    ) -> "EngineConfig":
        """Create a config using Core defaults plus caller-provided overrides."""
        return cls(
            artifact_root=(
                artifact_root if artifact_root is not None else resolve_default_artifact_root()
            ),
            provider_credentials=provider_credentials or ProviderCredentials(),
            prompt_override=prompt_override,
            default_soundfont_path=default_soundfont_path,
            max_generations=max_generations,
        )


@dataclass(frozen=True)
class GenerationRequest:
    """One prompt-to-loop generation request."""

    key: str
    scale: str
    description: str
    model: str
    provider: str | None = None
    temperature: float = 0.0
    use_thinking: bool = False
    effort: str = "low"
    prompt_override: str | None = None
    render_audio: bool = False
    soundfont_path: str | Path | None = None

    def __post_init__(self) -> None:
        if self.provider is not None:
            warnings.warn(
                "GenerationRequest.provider is deprecated and ignored; "
                "the provider is derived from the route actually used.",
                DeprecationWarning,
                stacklevel=2,
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
