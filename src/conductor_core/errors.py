"""Public exceptions for normalized provider failures."""


class ProviderError(RuntimeError):
    """Base error raised when a provider SDK fails.

    Attributes:
        provider: Display name of the provider that failed.
        operation: Provider operation that failed, when known.
    """

    def __init__(self, provider: str, message: str, *, operation: str | None = None) -> None:
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
