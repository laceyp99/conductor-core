import os
from pathlib import Path
from typing import get_type_hints
from unittest.mock import Mock

import pytest

from conductor_core import GenerationMetadata, GenerationRequest, GenerationResult
from conductor_core.models import Loop


def make_request(**overrides):
    values = {
        "key": "C",
        "scale": "Major",
        "description": "warm rhodes loop",
        "model": "gpt-4o-mini",
    }
    values.update(overrides)
    return GenerationRequest(**values)


def test_generation_result_exposes_concrete_public_types():
    type_hints = get_type_hints(GenerationResult)

    assert type_hints["loop"] is Loop
    assert type_hints["metadata"] is GenerationMetadata


def test_generation_request_rejects_removed_provider():
    with pytest.raises(TypeError, match="unexpected keyword argument 'provider'"):
        make_request(provider="Anthropic")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("key", None),
        ("key", 1),
        ("scale", None),
        ("scale", ["Major"]),
        ("description", None),
        ("description", 1),
        ("model", None),
        ("model", object()),
        ("temperature", "0.5"),
        ("temperature", True),
        ("use_thinking", "false"),
        ("use_thinking", 0),
        ("use_thinking", None),
        ("effort", 1),
        ("effort", False),
        ("prompt_override", 1),
        ("render_audio", "false"),
        ("render_audio", 0),
        ("render_audio", None),
        ("soundfont_path", 1),
        ("soundfont_path", b"soundfont.sf2"),
    ],
)
def test_generation_request_rejects_wrong_field_types(field_name, value):
    with pytest.raises(TypeError, match=rf"Invalid {field_name}"):
        make_request(**{field_name: value})


@pytest.mark.parametrize("key", ["H", "Cbbb", "c"])
def test_generation_request_rejects_unknown_keys(key):
    with pytest.raises(
        ValueError,
        match=rf"Invalid key {key!r}\. Expected one of: B#, C, Dbb",
    ):
        make_request(key=key)


@pytest.mark.parametrize("scale", ["major", "MINOR", "Harmonic Minor", "melodic minor"])
def test_generation_request_accepts_known_scales_case_insensitively(scale):
    assert make_request(scale=scale).scale == scale


@pytest.mark.parametrize("scale", ["dorian", ""])
def test_generation_request_rejects_unknown_scales(scale):
    with pytest.raises(ValueError, match="Invalid scale"):
        make_request(scale=scale)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("description", ""),
        ("description", " \t"),
        ("model", ""),
        ("model", " \n"),
        ("prompt_override", ""),
        ("prompt_override", "   "),
        ("soundfont_path", ""),
        ("soundfont_path", "   "),
    ],
)
def test_generation_request_rejects_blank_strings(field_name, value):
    with pytest.raises(ValueError, match=rf"Invalid {field_name}"):
        make_request(**{field_name: value})


@pytest.mark.parametrize("temperature", [-0.1, 2.1, float("inf"), float("nan")])
def test_generation_request_rejects_invalid_temperature_values(temperature):
    with pytest.raises(ValueError, match="Invalid temperature"):
        make_request(temperature=temperature)


@pytest.mark.parametrize("temperature", [0.0, 0.5, 1, 2.0])
def test_generation_request_accepts_temperature_in_range(temperature):
    assert make_request(temperature=temperature).temperature == temperature


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("description", "  warm loop  "),
        ("model", "  local model  "),
        ("prompt_override", "  custom prompt  "),
        ("soundfont_path", "  custom.sf2  "),
    ],
)
def test_generation_request_preserves_nonblank_strings(field_name, value):
    assert getattr(make_request(**{field_name: value}), field_name) == value


def test_generation_request_defaults_effort_to_none():
    assert make_request().effort is None
    assert make_request(effort=None).effort is None
    assert make_request(effort="provider-defined").effort == "provider-defined"


def test_invalid_request_fails_before_engine_can_be_invoked():
    generate = Mock()

    def construct_and_generate():
        generate(make_request(render_audio="false"))

    with pytest.raises(TypeError, match="Invalid render_audio"):
        construct_and_generate()

    generate.assert_not_called()


class DeferredPathLike(os.PathLike):
    def __fspath__(self):
        raise AssertionError("PathLike must not be evaluated during construction")


@pytest.mark.parametrize("soundfont_path", [Path("custom.sf2"), DeferredPathLike()])
def test_generation_request_accepts_pathlike_without_evaluating_it(soundfont_path):
    assert make_request(soundfont_path=soundfont_path).soundfont_path is soundfont_path
