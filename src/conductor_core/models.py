"""Models used to generate MIDI information."""

from enum import IntEnum

from pydantic import BaseModel, Field, model_validator

SIXTEENTHS_PER_BAR = 16
BARS_PER_LOOP = 4
SIXTEENTHS_PER_LOOP = SIXTEENTHS_PER_BAR * BARS_PER_LOOP

# Durations are deliberately a different vocabulary from positions.  A note may
# start only within its bar (1-16), but it may sustain for any part of the
# four-bar loop (1-64).
_DURATION_SIXTEENTH_NAMES = (
    "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE THIRTEEN "
    "FOURTEEN FIFTEEN SIXTEEN SEVENTEEN EIGHTEEN NINETEEN TWENTY TWENTY_ONE "
    "TWENTY_TWO TWENTY_THREE TWENTY_FOUR TWENTY_FIVE TWENTY_SIX TWENTY_SEVEN "
    "TWENTY_EIGHT TWENTY_NINE THIRTY THIRTY_ONE THIRTY_TWO THIRTY_THREE "
    "THIRTY_FOUR THIRTY_FIVE THIRTY_SIX THIRTY_SEVEN THIRTY_EIGHT THIRTY_NINE "
    "FORTY FORTY_ONE FORTY_TWO FORTY_THREE FORTY_FOUR FORTY_FIVE FORTY_SIX "
    "FORTY_SEVEN FORTY_EIGHT FORTY_NINE FIFTY FIFTY_ONE FIFTY_TWO FIFTY_THREE "
    "FIFTY_FOUR FIFTY_FIVE FIFTY_SIX FIFTY_SEVEN FIFTY_EIGHT FIFTY_NINE SIXTY "
    "SIXTY_ONE SIXTY_TWO SIXTY_THREE SIXTY_FOUR"
)


# Sixteenth Note Objects
class SixteenthNote(IntEnum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    ELEVEN = 11
    TWELVE = 12
    THIRTEEN = 13
    FOURTEEN = 14
    FIFTEEN = 15
    SIXTEEN = 16


DurationSixteenth = IntEnum(
    "DurationSixteenth",
    _DURATION_SIXTEENTH_NAMES,
    start=1,
)


# Time Information Objects
class TimeInformation(BaseModel):
    start_beat: SixteenthNote = Field(
        ...,
        description="Starting beat of the note in sixteenth notes (e.g. 1-16). REMEMBER THIS IS BASE 1 NOT 0.",
    )
    duration: DurationSixteenth = Field(
        ...,
        description="Duration in sixteenth notes (1-64 for the four-bar loop).",
    )


# Note Objects
def _validate_midi_pitch(note):
    """Ensure a pitch and octave identify an encodable MIDI note."""
    from conductor_core.music import calculate_midi_number

    midi_number = calculate_midi_number(note)
    if not 0 <= midi_number <= 127:
        raise ValueError(
            f"Pitch {note.pitch}{note.octave} maps to MIDI note {midi_number}; "
            "expected a value from 0 to 127"
        )
    return note


class Note(BaseModel):
    pitch: str = Field(
        ...,
        description='Pitch of the note (e.g. "C", "D", "E", "F", "G", "A", "B") Please do not include the octave number',
    )
    octave: int = Field(
        ...,
        strict=True,
        ge=-2,
        le=9,
        description=(
            "Scientific pitch octave from -2 through 9; the exact MIDI range "
            "depends on the pitch spelling"
        ),
    )
    velocity: int = Field(
        ...,
        strict=True,
        ge=1,
        le=127,
        description="Note-on velocity (1-127)",
    )
    time: TimeInformation

    @model_validator(mode="after")
    def validate_midi_pitch(self):
        return _validate_midi_pitch(self)


# Bar Objects
class Bar(BaseModel):
    num: int = Field(..., description="Number of the bar (e.g. 1-4)")
    notes: list[Note] = Field(..., description="List of notes in the bar")


# Loop Objects
def _validate_loop_note_boundaries(loop: "Loop"):
    """Reject notes whose sustain would extend beyond the four-bar loop."""
    for bar_index in range(BARS_PER_LOOP):
        bar = getattr(loop, f"Bar_{bar_index + 1}")
        for note in bar.notes:
            start = note.time.start_beat
            duration = note.time.duration
            if (
                bar_index * SIXTEENTHS_PER_BAR + (start - 1) + duration
                > SIXTEENTHS_PER_LOOP
            ):
                raise ValueError(
                    "Note duration extends beyond the four-bar loop boundary. "
                    "Shorten the duration or start the note earlier."
                )
    return loop


class Loop(BaseModel):
    Bar_1: Bar = Field(..., description="The first bar of the four bar loop")
    Bar_2: Bar = Field(..., description="The second bar of the four bar loop")
    Bar_3: Bar = Field(..., description="The third bar of the four bar loop")
    Bar_4: Bar = Field(..., description="The fourth bar of the four bar loop")

    @model_validator(mode="after")
    def validate_note_boundaries(self):
        return _validate_loop_note_boundaries(self)
