import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS_ROOT = Path(__file__).parents[1] / "scripts"


def load_script(name):
    script_path = SCRIPTS_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"example_{name}", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name",
    ["generate_midi", "midi_loop_roundtrip", "inspect_models"],
)
def test_example_script_imports_without_running_main(name, monkeypatch):
    monkeypatch.setattr("conductor_core.engine.LoopGenerationEngine.generate", lambda *args: None)
    module = load_script(name)

    assert callable(module.main)


def test_generate_midi_builds_expected_paid_request(capsys):
    script = load_script("generate_midi")
    captured = {}
    result = SimpleNamespace(
        generation_id="example-id",
        midi_path="generations/example/loop.mid",
        audio_path="generations/example/loop.mp3",
        cost=0.42,
        warnings=[],
    )

    class FakeEngine:
        def generate(self, request, progress_callback=None):
            captured["request"] = request
            progress_callback(
                SimpleNamespace(stage="provider_call", message="Generating MIDI...", detail=None)
            )
            return result

    assert script.main(engine=FakeEngine()) is result

    request = captured["request"]
    assert request.model == "gpt-5.6-terra"
    assert request.description == (
        "warm neo-soul electric piano chords with syncopated upper extensions "
        "and a simple bass movement"
    )
    assert request.effort == "medium"
    assert request.prompt_override is None
    assert request.render_audio is True
    assert request.soundfont_path is None
    output = capsys.readouterr().out
    assert "[provider_call] Generating MIDI..." in output
    assert "generations/example/loop.mp3" in output


def test_midi_roundtrip_uses_sixteenth_note_integer_timing(
    tmp_path,
    sample_loop,
    midi_builder,
):
    script = load_script("midi_loop_roundtrip")
    input_path = midi_builder(sample_loop, times_as_string=False, filename="source.mid")
    output_path = tmp_path / "nested" / "roundtrip.mid"

    restored = script.roundtrip_midi(input_path, output_path)

    assert output_path.is_file()
    bars = [restored.Bar_1, restored.Bar_2, restored.Bar_3, restored.Bar_4]
    assert [bar.notes[0].pitch for bar in bars] == ["C", "E", "G", "B"]
    assert all(bar.notes[0].time.start_beat == 1 for bar in bars)
    assert all(bar.notes[0].time.duration == 16 for bar in bars)


def test_midi_roundtrip_refuses_to_overwrite_input(tmp_path):
    script = load_script("midi_loop_roundtrip")
    midi_path = tmp_path / "loop.mid"

    with pytest.raises(ValueError, match="must be different"):
        script.roundtrip_midi(midi_path, midi_path)


def test_model_discovery_formats_packaged_metadata_without_provider_calls(capsys):
    script = load_script("inspect_models")
    model_info = {
        "models": {
            "Example Provider": {
                "example-model": {
                    "extended_thinking": True,
                    "effort_options": ["low", "medium"],
                    "cost": {"input": 1.0},
                }
            }
        }
    }
    script.get_model_info = lambda: model_info

    assert script.main() is model_info

    output = capsys.readouterr().out
    assert "Example Provider" in output
    assert "example-model" in output
    assert "extended thinking: True" in output
    assert "effort options: low, medium" in output
    assert "cost" not in output
