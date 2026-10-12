"""Shared discrete-effort validation for direct hosted-provider calls."""


def _resolve_discrete_effort(
    model: str,
    effort_options: list[str],
    use_thinking: bool | None,
    effort: str | None,
) -> str | None:
    """Normalize disabled thinking and reject unsupported enabled effort."""
    if effort_options and not use_thinking:
        return effort_options[0]
    if effort_options and effort not in effort_options:
        supported_values = ", ".join(effort_options)
        raise ValueError(
            f"Invalid effort {effort!r} for {model}. Expected one of: {supported_values}"
        )
    return effort
