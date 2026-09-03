import pytest
from pydantic import ValidationError

from conductor_core.models import (
    DurationSixteenth,
    Loop,
    Note,
    SixteenthNote,
    TimeInformation,
)


def test_time_information_coerces_integer_inputs_to_enum_members():
    time_info = TimeInformation(start_beat=1, duration=16)

    assert time_info.start_beat is SixteenthNote.ONE
    assert time_info.duration is DurationSixteenth.SIXTEEN


def test_duration_has_a_dedicated_four_bar_vocabulary():
    time_info = TimeInformation(start_beat=16, duration=17)

    assert time_info.start_beat is SixteenthNote.SIXTEEN
    assert time_info.duration is DurationSixteenth.SEVENTEEN


def test_note_accepts_midi_pitch_boundaries():
    time = {"start_beat": 1, "duration": 1}
    low_note = Note(pitch="C", octave=-1, velocity=1, time=time)
    low_enharmonic = Note(pitch="B#", octave=-2, velocity=1, time=time)
    low_double_sharp = Note(pitch="B##", octave=-2, velocity=1, time=time)
    high_note = Note(pitch="G", octave=9, velocity=127, time=time)

    assert (low_note.octave, low_note.velocity) == (-1, 1)
    assert (low_enharmonic.octave, low_enharmonic.velocity) == (-2, 1)
    assert (low_double_sharp.octave, low_double_sharp.velocity) == (-2, 1)
    assert (high_note.octave, high_note.velocity) == (9, 127)


@pytest.mark.parametrize(
    ("pitch", "octave", "velocity", "message"),
    [
        ("C", -2, 96, "maps to MIDI note -12"),
        ("B", -3, 96, "greater than or equal to -2"),
        ("G#", 9, 96, "maps to MIDI note 128"),
        ("C", 10, 96, "less than or equal to 9"),
        ("C", 4, 0, "greater than or equal to 1"),
        ("C", 4, 128, "less than or equal to 127"),
        ("H", 4, 96, "Unrecognized note name"),
    ],
)
def test_note_rejects_values_that_cannot_create_note_on_events(
    pitch, octave, velocity, message
):
    with pytest.raises(ValidationError, match=message):
        Note(
            pitch=pitch,
            octave=octave,
            velocity=velocity,
            time={"start_beat": 1, "duration": 1},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("octave", True),
        ("octave", 4.0),
        ("octave", "4"),
        ("velocity", True),
        ("velocity", 96.0),
        ("velocity", "96"),
    ],
)
def test_note_requires_integer_octave_and_velocity(field, value):
    values = {
        "pitch": "C",
        "octave": 4,
        "velocity": 96,
        "time": {"start_beat": 1, "duration": 1},
    }
    values[field] = value

    with pytest.raises(ValidationError, match="Input should be a valid integer"):
        Note(**values)


def test_loop_validates_nested_bar_and_note_data_from_dicts():
    loop = Loop(
        Bar_1={
            "num": 1,
            "notes": [
                {
                    "pitch": "C",
                    "octave": 4,
                    "velocity": 96,
                    "time": {"start_beat": 1, "duration": 16},
                }
            ],
        },
        Bar_2={"num": 2, "notes": []},
        Bar_3={"num": 3, "notes": []},
        Bar_4={"num": 4, "notes": []},
    )

    assert loop.Bar_1.notes[0].time.start_beat is SixteenthNote.ONE
    assert loop.Bar_1.notes[0].time.duration is DurationSixteenth.SIXTEEN


def test_loop_rejects_notes_extending_beyond_its_four_bar_boundary():
    bar = {"num": 1, "notes": []}
    final_bar = {
        "num": 4,
        "notes": [
            {
                "pitch": "C",
                "octave": 4,
                "velocity": 96,
                "time": {"start_beat": 16, "duration": 2},
            }
        ],
    }

    with pytest.raises(ValidationError, match="four-bar loop boundary"):
        Loop(
            Bar_1=bar, Bar_2={**bar, "num": 2}, Bar_3={**bar, "num": 3}, Bar_4=final_bar
        )


def test_loop_schema_distinguishes_duration_from_start_position():
    schema = Loop.model_json_schema()
    definitions = schema["$defs"]

    assert 64 in definitions["DurationSixteenth"]["enum"]
    assert definitions["SixteenthNote"]["enum"] == list(range(1, 17))
    assert definitions["Note"]["properties"]["octave"]["minimum"] == -2
    assert definitions["Note"]["properties"]["octave"]["maximum"] == 9
    assert definitions["Note"]["properties"]["velocity"]["minimum"] == 1
    assert definitions["Note"]["properties"]["velocity"]["maximum"] == 127
