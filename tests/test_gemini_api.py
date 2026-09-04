from types import SimpleNamespace

import pytest

from conductor_core.providers import google as gemini_api


def test_calc_cost_uses_reported_cached_tokens_without_storage_estimate():
    usage = SimpleNamespace(
        prompt_token_count=1000,
        candidates_token_count=200,
        thoughts_token_count=None,
        cached_content_token_count=400,
    )

    cost = gemini_api.calc_cost("gemini-2.5-pro", usage)

    expected = (
        (600 * 1.25 / 1_000_000) + (200 * 10.00 / 1_000_000) + (400 * 0.125 / 1_000_000)
    )
    assert cost == pytest.approx(expected)


def test_calc_cost_uses_standard_rate_above_legacy_threshold():
    usage = SimpleNamespace(
        prompt_token_count=250000,
        candidates_token_count=100,
        thoughts_token_count=None,
        cached_content_token_count=0,
    )

    cost = gemini_api.calc_cost("gemini-3.1-pro-preview", usage)

    expected = (250000 * 2.00 + 100 * 12.00) / 1_000_000
    assert cost == pytest.approx(expected)


def test_calc_cost_includes_thought_tokens_at_output_rate():
    usage = SimpleNamespace(
        prompt_token_count=1000,
        candidates_token_count=200,
        thoughts_token_count=300,
        cached_content_token_count=400,
    )

    cost = gemini_api.calc_cost("gemini-2.5-flash", usage)

    expected = (
        (600 * 0.30 / 1_000_000)
        + ((200 + 300) * 2.50 / 1_000_000)
        + (400 * 0.03 / 1_000_000)
    )
    assert cost == pytest.approx(expected)


@pytest.mark.parametrize("thoughts_token_count", [None, 0])
def test_calc_cost_treats_empty_thought_count_as_zero(thoughts_token_count):
    usage = SimpleNamespace(
        prompt_token_count=1000,
        candidates_token_count=200,
        thoughts_token_count=thoughts_token_count,
        cached_content_token_count=400,
    )

    cost = gemini_api.calc_cost("gemini-2.5-flash", usage)

    expected = (
        (600 * 0.30 / 1_000_000) + (200 * 2.50 / 1_000_000) + (400 * 0.03 / 1_000_000)
    )
    assert cost == pytest.approx(expected)


def test_calc_cost_clamps_cached_tokens_to_avoid_negative_input_cost():
    usage = SimpleNamespace(
        prompt_token_count=100,
        candidates_token_count=0,
        thoughts_token_count=None,
        cached_content_token_count=150,
    )

    cost = gemini_api.calc_cost("gemini-2.5-pro", usage)

    assert cost == pytest.approx(100 * 0.125 / 1_000_000)


def test_calc_cost_handles_models_without_cache_pricing():
    usage = SimpleNamespace(
        prompt_token_count=1000,
        candidates_token_count=200,
        thoughts_token_count=None,
        cached_content_token_count=None,
    )

    cost = gemini_api.calc_cost("gemini-2.5-flash-lite", usage)

    expected = (1000 * 0.10 / 1_000_000) + (200 * 0.40 / 1_000_000)
    assert cost == pytest.approx(expected)


@pytest.mark.parametrize("model", ["gemini-2.5-flash", "gemini-2.5-pro"])
def test_calc_cost_treats_missing_token_counts_as_zero(model):
    usage = SimpleNamespace(
        prompt_token_count=None,
        candidates_token_count=None,
        thoughts_token_count=None,
        cached_content_token_count=None,
    )

    assert gemini_api.calc_cost(model, usage) == 0
