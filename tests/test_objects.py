import pytest
from pydantic import ValidationError

from conductor_core.models import (
    DurationSixteenth,
    DurationSixteenth_G,
    Loop,
    Loop_G,
    SixteenthNote,
    SixteenthNote_G,
    TimeInformation,
    TimeInformation_G,
)


def test_sixteenth_note_g_from_int_returns_expected_enum_member():
    assert SixteenthNote_G.from_int(16) is SixteenthNote_G.SIXTEEN


def test_sixteenth_note_g_from_int_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="Invalid sixteenth-note value"):
        SixteenthNote_G.from_int(17)


def test_time_information_coerces_integer_inputs_to_enum_members():
    time_info = TimeInformation(start_beat=1, duration=16)

    assert time_info.start_beat is SixteenthNote.ONE
    assert time_info.duration is DurationSixteenth.SIXTEEN


def test_duration_has_a_dedicated_four_bar_vocabulary():
    time_info = TimeInformation(start_beat=16, duration=17)

    assert time_info.start_beat is SixteenthNote.SIXTEEN
    assert time_info.duration is DurationSixteenth.SEVENTEEN
    assert TimeInformation_G(start_beat="one", duration="sixty_four").duration is (
        DurationSixteenth_G.SIXTY_FOUR
    )


def test_time_information_g_rejects_unknown_string_values():
    with pytest.raises(ValidationError):
        TimeInformation_G(start_beat="zero", duration="sixteen")


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


@pytest.mark.parametrize(
    ("payload", "loop_type"),
    [
        (
            {"start_beat": 16, "duration": 2},
            Loop,
        ),
        (
            {"start_beat": "sixteen", "duration": "two"},
            Loop_G,
        ),
    ],
)
def test_loop_rejects_notes_extending_beyond_its_four_bar_boundary(payload, loop_type):
    bar = {"num": 1, "notes": []}
    final_bar = {"num": 4, "notes": [{"pitch": "C", "octave": 4, "velocity": 96, "time": payload}]}

    with pytest.raises(ValidationError, match="four-bar loop boundary"):
        loop_type(Bar_1=bar, Bar_2={**bar, "num": 2}, Bar_3={**bar, "num": 3}, Bar_4=final_bar)


def test_loop_schema_distinguishes_duration_from_start_position():
    schema = Loop.model_json_schema()
    definitions = schema["$defs"]

    assert 64 in definitions["DurationSixteenth"]["enum"]
    assert definitions["SixteenthNote"]["enum"] == list(range(1, 17))

    gemini_definitions = Loop_G.model_json_schema()["$defs"]
    assert "sixty_four" in gemini_definitions["DurationSixteenth_G"]["enum"]
    assert gemini_definitions["SixteenthNote_G"]["enum"] == [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
    ]
