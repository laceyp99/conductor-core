"""Public exceptions raised by Conductor Core."""


class AudioRenderingError(RuntimeError):
    """Core could not render a MIDI file to audio."""


class ProviderError(RuntimeError):
    """Base error raised when a provider SDK fails.

    Attributes:
        provider: Display name of the provider that failed.
        operation: Provider operation that failed, when known.
    """

    def __init__(
        self, provider: str, message: str, *, operation: str | None = None
    ) -> None:
        self.provider = provider
        self.operation = operation
        context = f" {operation}" if operation else ""
        super().__init__(f"{provider}{context} failed: {message}")


class ProviderAuthenticationError(ProviderError):
    """Provider credentials are missing or were rejected."""


class ProviderRateLimitError(ProviderError):
    """The provider rejected a request because of a rate limit."""


class ProviderConnectionError(ProviderError):
    """Core could not establish or maintain a connection to a provider."""


class ProviderTimeoutError(ProviderConnectionError):
    """A provider operation exceeded its configured or SDK-default timeout."""


class ProviderRequestError(ProviderError):
    """A provider rejected or failed while processing a request."""


class ProviderContextLengthError(ProviderRequestError):
    """A response filled the model's context window before it was complete.

    Attributes:
        model: Model that ran out of context.
        prompt_tokens: Prompt tokens the provider reported, when known.
        output_tokens: Generated tokens, including thinking, when known.
        context_length: Context window size Core requested, when known.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        prompt_tokens: int | None = None,
        output_tokens: int | None = None,
        context_length: int | None = None,
        operation: str | None = None,
    ) -> None:
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.context_length = context_length
        usage = []
        if prompt_tokens is not None:
            usage.append(f"prompt ({prompt_tokens:,} tokens)")
        if output_tokens is not None:
            usage.append(f"response ({output_tokens:,} tokens)")
        usage_text = " and ".join(usage) if usage else "prompt and response"
        window = (
            f"{context_length:,}-token context window"
            if context_length is not None
            else "context window"
        )
        super().__init__(
            provider,
            f"model {model!r} ran out of context: the {usage_text} filled the "
            f"{window} before the answer was complete. Disable thinking, use a "
            "model or server with a larger context, or request a larger context "
            "size.",
            operation=operation,
        )


def error_for_status(
    provider: str,
    message: str,
    status_code: int | None,
    *,
    operation: str,
) -> ProviderError:
    """Return the normalized error type for an HTTP-like provider status."""
    if status_code in {401, 403}:
        return ProviderAuthenticationError(provider, message, operation=operation)
    if status_code == 429:
        return ProviderRateLimitError(provider, message, operation=operation)
    if status_code in {408, 504}:
        return ProviderTimeoutError(provider, message, operation=operation)
    return ProviderRequestError(provider, message, operation=operation)
