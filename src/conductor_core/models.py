"""
This file holds all the objects that will be used to generate MIDI information.

Note: Gemini structured outputs do not support integer enums. To work around this limitation, _G objects are used instead.
"""

from enum import Enum, IntEnum

from pydantic import BaseModel, Field, model_validator

SIXTEENTHS_PER_BAR = 16
BARS_PER_LOOP = 4
SIXTEENTHS_PER_LOOP = SIXTEENTHS_PER_BAR * BARS_PER_LOOP

SIXTEENTH_NOTE_G_TO_INT = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
}
SIXTEENTH_NOTE_INT_TO_G = {value: key for key, value in SIXTEENTH_NOTE_G_TO_INT.items()}

# Durations are deliberately a different vocabulary from positions.  A note may
# start only within its bar (1-16), but it may sustain for any part of the
# four-bar loop (1-64).  Gemini only accepts string enums in response schemas,
# so its duration vocabulary mirrors the integer duration values as words.
DURATION_SIXTEENTH_G_TO_INT = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty_one": 21,
    "twenty_two": 22,
    "twenty_three": 23,
    "twenty_four": 24,
    "twenty_five": 25,
    "twenty_six": 26,
    "twenty_seven": 27,
    "twenty_eight": 28,
    "twenty_nine": 29,
    "thirty": 30,
    "thirty_one": 31,
    "thirty_two": 32,
    "thirty_three": 33,
    "thirty_four": 34,
    "thirty_five": 35,
    "thirty_six": 36,
    "thirty_seven": 37,
    "thirty_eight": 38,
    "thirty_nine": 39,
    "forty": 40,
    "forty_one": 41,
    "forty_two": 42,
    "forty_three": 43,
    "forty_four": 44,
    "forty_five": 45,
    "forty_six": 46,
    "forty_seven": 47,
    "forty_eight": 48,
    "forty_nine": 49,
    "fifty": 50,
    "fifty_one": 51,
    "fifty_two": 52,
    "fifty_three": 53,
    "fifty_four": 54,
    "fifty_five": 55,
    "fifty_six": 56,
    "fifty_seven": 57,
    "fifty_eight": 58,
    "fifty_nine": 59,
    "sixty": 60,
    "sixty_one": 61,
    "sixty_two": 62,
    "sixty_three": 63,
    "sixty_four": 64,
}
DURATION_SIXTEENTH_INT_TO_G = {value: key for key, value in DURATION_SIXTEENTH_G_TO_INT.items()}


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


class SixteenthNote_G(Enum):
    ONE = "one"
    TWO = "two"
    THREE = "three"
    FOUR = "four"
    FIVE = "five"
    SIX = "six"
    SEVEN = "seven"
    EIGHT = "eight"
    NINE = "nine"
    TEN = "ten"
    ELEVEN = "eleven"
    TWELVE = "twelve"
    THIRTEEN = "thirteen"
    FOURTEEN = "fourteen"
    FIFTEEN = "fifteen"
    SIXTEEN = "sixteen"

    @classmethod
    def from_int(cls, value: int) -> "SixteenthNote_G":
        """Create a Gemini-compatible sixteenth-note enum from an integer."""
        try:
            return cls(SIXTEENTH_NOTE_INT_TO_G[value])
        except KeyError as exc:
            raise ValueError(f"Invalid sixteenth-note value: {value}") from exc


DurationSixteenth = IntEnum(
    "DurationSixteenth",
    {name.upper(): value for name, value in DURATION_SIXTEENTH_G_TO_INT.items()},
)
DurationSixteenth_G = Enum(
    "DurationSixteenth_G",
    {name.upper(): name for name in DURATION_SIXTEENTH_G_TO_INT},
    type=str,
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


class TimeInformation_G(BaseModel):
    start_beat: SixteenthNote_G = Field(
        ...,
        description="Starting beat of the note in sixteenth notes (e.g. 1-16). REMEMBER THIS IS BASE 1 NOT 0.",
    )
    duration: DurationSixteenth_G = Field(
        ...,
        description="Duration in sixteenth notes (one-sixty_four for the four-bar loop).",
    )


# Note Objects
class Note(BaseModel):
    pitch: str = Field(
        ...,
        description='Pitch of the note (e.g. "C", "D", "E", "F", "G", "A", "B") Please do not include the octave number',
    )
    octave: int = Field(..., description="Octave of the note (e.g. 1-8)")
    velocity: int = Field(..., description="Velocity of the note (e.g. 0-127)")
    time: TimeInformation


class Note_G(BaseModel):
    pitch: str = Field(
        ...,
        description='Pitch of the note (e.g. "C", "D", "E", "F", "G", "A", "B") Please do not include the octave number',
    )
    octave: int = Field(..., description="Octave of the note (e.g. 1-8)")
    velocity: int = Field(..., description="Velocity of the note (e.g. 0-127)")
    time: TimeInformation_G = Field(..., description="Time information of the note")


# Bar Objects
class Bar(BaseModel):
    num: int = Field(..., description="Number of the bar (e.g. 1-4)")
    notes: list[Note] = Field(..., description="List of notes in the bar")


class Bar_G(BaseModel):
    num: int = Field(..., description="Number of the bar (e.g. 1-4)")
    notes: list[Note_G] = Field(..., description="List of notes in the bar")


# Loop Objects
def _validate_loop_note_boundaries(loop: "Loop | Loop_G"):
    """Reject notes whose sustain would extend beyond the four-bar loop."""
    for bar_index in range(BARS_PER_LOOP):
        bar = getattr(loop, f"Bar_{bar_index + 1}")
        for note in bar.notes:
            start_value = getattr(note.time.start_beat, "value", note.time.start_beat)
            duration_value = getattr(note.time.duration, "value", note.time.duration)
            start = SIXTEENTH_NOTE_G_TO_INT.get(start_value, start_value)
            duration = DURATION_SIXTEENTH_G_TO_INT.get(duration_value, duration_value)
            if bar_index * SIXTEENTHS_PER_BAR + (start - 1) + duration > SIXTEENTHS_PER_LOOP:
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


class Loop_G(BaseModel):
    Bar_1: Bar_G = Field(..., description="The first bar of the four bar loop")
    Bar_2: Bar_G = Field(..., description="The second bar of the four bar loop")
    Bar_3: Bar_G = Field(..., description="The third bar of the four bar loop")
    Bar_4: Bar_G = Field(..., description="The fourth bar of the four bar loop")

    @model_validator(mode="after")
    def validate_note_boundaries(self):
        return _validate_loop_note_boundaries(self)
