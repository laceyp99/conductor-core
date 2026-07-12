"""Inspect Conductor Core's packaged model metadata without network access.

Run ``python scripts/inspect_models.py`` from the repository root. This script
does not contact OpenAI, Anthropic, Google, Ollama, or any other service.
"""

from conductor_core.music import get_model_info


def format_capability(name, value):
    """Format one metadata capability for compact terminal output."""
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    elif isinstance(value, dict):
        value = ", ".join(f"{key}={item}" for key, item in value.items())
    return f"{name.replace('_', ' ')}: {value}"


def iter_model_lines(model_info):
    """Yield readable provider, model, and capability lines."""
    for provider, models in model_info.get("models", {}).items():
        yield provider
        for model_name, capabilities in models.items():
            details = "; ".join(
                format_capability(name, value)
                for name, value in capabilities.items()
                if name not in {"cost", "rate_limits"}
            )
            suffix = f" — {details}" if details else ""
            yield f"  - {model_name}{suffix}"


def main():
    """Print the packaged model catalog and return it."""
    model_info = get_model_info()
    print("Packaged models (offline metadata):")
    for line in iter_model_lines(model_info):
        print(line)
    return model_info


if __name__ == "__main__":
    main()
