from pathlib import Path

import pytest
from mido import MidiFile

from conductor_core.midi import loop_to_midi
from conductor_core.models import (
    Bar,
    Loop,
    Note,
    TimeInformation,
)


@pytest.fixture
def note_factory():
    def factory(
        *,
        pitch="C",
        octave=4,
        velocity=96,
        start_beat=1,
        duration=16,
    ):
        return Note(
            pitch=pitch,
            octave=octave,
            velocity=velocity,
            time=TimeInformation(start_beat=start_beat, duration=duration),
        )

    return factory


@pytest.fixture
def loop_factory(note_factory):
    def factory(*, bars=None):
        if bars is None:
            bars = [
                [note_factory(pitch="C", start_beat=1, duration=16)],
                [note_factory(pitch="E", start_beat=1, duration=16)],
                [note_factory(pitch="G", start_beat=1, duration=16)],
                [note_factory(pitch="B", start_beat=1, duration=16)],
            ]

        return Loop(
            Bar_1=Bar(num=1, notes=bars[0]),
            Bar_2=Bar(num=2, notes=bars[1]),
            Bar_3=Bar(num=3, notes=bars[2]),
            Bar_4=Bar(num=4, notes=bars[3]),
        )

    return factory


@pytest.fixture
def sample_loop(loop_factory):
    return loop_factory()


@pytest.fixture
def midi_builder(tmp_path):
    def factory(loop, *, ticks_per_beat=480, filename="test_loop.mid"):
        midi = MidiFile(ticks_per_beat=ticks_per_beat)
        loop_to_midi(midi, loop)

        midi_path = Path(tmp_path) / filename
        midi.save(midi_path)
        return midi_path

    return factory
