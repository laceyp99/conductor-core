"""Verify the installed Google extra without other provider dependencies."""

import json
import warnings
from importlib.util import find_spec
from unittest.mock import patch

import httpx2
from google import genai
from google.genai import types

from conductor_core import ProviderTimeoutError
from conductor_core.providers import google

assert find_spec("httpx") is None, "Run this check in a fresh Google-only environment"
assert google.genai is not None, "Google extra failed to enable the adapter"

calls = []


def respond(request):
    body = json.loads(request.content)
    thinking = body["generationConfig"]["thinkingConfig"]
    assert thinking.get("thinkingLevel", thinking.get("thinking_level")) == "XHIGH"
    assert "temperature" not in body["generationConfig"]
    calls.append(body)
    raise httpx2.ReadTimeout("offline timeout", request=request)


with warnings.catch_warnings():
    warnings.simplefilter("error", UserWarning)
    with (
        genai.Client(
            api_key="offline-test",
            http_options=types.HttpOptions(
                client_args={"transport": httpx2.MockTransport(respond)},
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        ) as client,
        patch.object(google, "initialize_gemini_client", return_value=client),
    ):
        for generate in (google.loop_gen, google.variations_gen):
            error = None
            try:
                generate(
                    "offline prompt",
                    "gemini-4-hydrogen",
                    use_thinking=True,
                    effort="xhigh",
                )
            except ProviderTimeoutError as exc:
                error = exc
            assert error is not None, "Google did not normalize the transport timeout"
            assert isinstance(error.__cause__, httpx2.ReadTimeout)

assert len(calls) == 2
print(
    "Google-only installed-wheel import, xhigh serialization, and timeout checks passed"
)
