import pytest
from mido import Message, MetaMessage, MidiFile, MidiTrack
from pydantic import ValidationError

from conductor_core.midi import loop_to_midi, midi_to_loop


def _note_events_with_absolute_times(midi):
    events = []
    absolute_time = 0
    for msg in midi.tracks[0]:
        absolute_time += msg.time
        if msg.type != "end_of_track":
            events.append((msg.type, msg.note, absolute_time))
    return events


def test_loop_to_midi_orders_note_off_before_note_on_at_same_tick(loop_factory, note_factory):
    loop = loop_factory(
        bars=[
            [
                note_factory(pitch="C", start_beat=1, duration=4),
                note_factory(pitch="E", start_beat=5, duration=4),
            ],
            [],
            [],
            [],
        ]
    )
    midi = MidiFile(ticks_per_beat=480)

    loop_to_midi(midi, loop, times_as_string=False)

    messages = [msg for msg in midi.tracks[0] if msg.type != "end_of_track"]

    assert [(msg.type, msg.note, msg.time) for msg in messages] == [
        ("note_on", 60, 0),
        ("note_off", 60, 480),
        ("note_on", 64, 0),
        ("note_off", 64, 480),
    ]


def test_loop_to_midi_clamps_out_of_range_velocity(loop_factory, note_factory):
    loop = loop_factory(
        bars=[
            [note_factory(pitch="C", start_beat=1, duration=4, velocity=200)],
            [],
            [],
            [],
        ]
    )
    midi = MidiFile(ticks_per_beat=480)

    loop_to_midi(midi, loop, times_as_string=False)

    note_messages = [msg for msg in midi.tracks[0] if msg.type != "end_of_track"]

    assert [msg.velocity for msg in note_messages] == [127, 127]


def test_loop_to_midi_allows_notes_to_cross_early_bar_boundaries(loop_factory, note_factory):
    loop = loop_factory(
        bars=[
            [note_factory(pitch="C", start_beat=16, duration=4)],
            [],
            [],
            [],
        ]
    )
    midi = MidiFile(ticks_per_beat=480)

    loop_to_midi(midi, loop, times_as_string=False)

    assert _note_events_with_absolute_times(midi) == [
        ("note_on", 60, 1800),
        ("note_off", 60, 2280),
    ]


def test_loop_rejects_notes_past_four_bar_boundary(loop_factory, note_factory):
    with pytest.raises(ValidationError, match="four-bar loop boundary"):
        loop_factory(bars=[[], [], [], [note_factory(pitch="C", start_beat=16, duration=4)]])


def test_loop_to_midi_preserves_note_at_exact_four_bar_boundary(loop_factory, note_factory):
    loop = loop_factory(bars=[[], [], [], [note_factory(pitch="C", start_beat=16, duration=1)]])
    midi = MidiFile(ticks_per_beat=480)

    loop_to_midi(midi, loop, times_as_string=False)

    assert _note_events_with_absolute_times(midi) == [
        ("note_on", 60, 7560),
        ("note_off", 60, 7680),
    ]


def test_midi_to_loop_round_trips_integer_timing(sample_loop, midi_builder):
    midi_path = midi_builder(sample_loop, times_as_string=False)

    loop = midi_to_loop(str(midi_path), times_as_string=False)

    assert [bar.notes[0].pitch for bar in [loop.Bar_1, loop.Bar_2, loop.Bar_3, loop.Bar_4]] == [
        "C",
        "E",
        "G",
        "B",
    ]
    assert all(
        bar.notes[0].time.start_beat == 1
        for bar in [loop.Bar_1, loop.Bar_2, loop.Bar_3, loop.Bar_4]
    )
    assert all(
        bar.notes[0].time.duration == 16 for bar in [loop.Bar_1, loop.Bar_2, loop.Bar_3, loop.Bar_4]
    )


def test_midi_to_loop_round_trips_string_timing(sample_loop_g, midi_builder):
    midi_path = midi_builder(sample_loop_g, times_as_string=True)

    loop = midi_to_loop(str(midi_path), times_as_string=True)

    assert [bar.notes[0].pitch for bar in [loop.Bar_1, loop.Bar_2, loop.Bar_3, loop.Bar_4]] == [
        "C",
        "E",
        "G",
        "B",
    ]
    assert all(
        bar.notes[0].time.start_beat.value == "one"
        for bar in [loop.Bar_1, loop.Bar_2, loop.Bar_3, loop.Bar_4]
    )
    assert all(
        bar.notes[0].time.duration.value == "sixteen"
        for bar in [loop.Bar_1, loop.Bar_2, loop.Bar_3, loop.Bar_4]
    )


def test_midi_to_loop_skips_notes_beyond_the_first_four_bars(tmp_path):
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    track.append(Message("note_on", note=60, velocity=96, time=7680))
    track.append(Message("note_off", note=60, velocity=96, time=120))
    track.append(MetaMessage("end_of_track", time=0))
    midi.tracks.append(track)

    midi_path = tmp_path / "fifth_bar.mid"
    midi.save(midi_path)

    loop = midi_to_loop(str(midi_path), times_as_string=False)

    assert loop.Bar_1.notes == []
    assert loop.Bar_2.notes == []
    assert loop.Bar_3.notes == []
    assert loop.Bar_4.notes == []


@pytest.mark.parametrize(
    ("times_as_string", "expected_duration"),
    [(False, 17), (True, "seventeen")],
)
def test_midi_to_loop_imports_seventeen_sixteenth_sustains(
    tmp_path, times_as_string, expected_duration
):
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack(
        [
            Message("note_on", note=60, velocity=96, time=0),
            Message("note_off", note=60, velocity=0, time=2040),
        ]
    )
    midi.tracks.append(track)
    path = tmp_path / "seventeen.mid"
    midi.save(path)

    loop = midi_to_loop(str(path), times_as_string=times_as_string)

    duration = loop.Bar_1.notes[0].time.duration
    assert duration.value == expected_duration if times_as_string else duration == expected_duration


@pytest.mark.parametrize("times_as_string", [False, True])
def test_midi_long_note_round_trips_across_multiple_bars(tmp_path, times_as_string):
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack(
        [
            Message("note_on", note=60, velocity=96, time=0),
            Message("note_off", note=60, velocity=0, time=5760),
        ]
    )
    midi.tracks.append(track)
    source = tmp_path / "three_bars.mid"
    midi.save(source)

    loop = midi_to_loop(str(source), times_as_string=times_as_string)
    exported = MidiFile(ticks_per_beat=480)
    loop_to_midi(exported, loop, times_as_string=times_as_string)

    assert _note_events_with_absolute_times(exported) == [
        ("note_on", 60, 0),
        ("note_off", 60, 5760),
    ]


def test_midi_to_loop_clips_note_at_exact_four_bar_boundary(tmp_path):
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack(
        [
            Message("note_on", note=60, velocity=96, time=7560),
            Message("note_off", note=60, velocity=0, time=480),
        ]
    )
    midi.tracks.append(track)
    path = tmp_path / "boundary.mid"
    midi.save(path)

    loop = midi_to_loop(str(path), times_as_string=False)

    assert loop.Bar_4.notes[0].time.start_beat == 16
    assert loop.Bar_4.notes[0].time.duration == 1
