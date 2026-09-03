import pytest
from mido import Message, MetaMessage, MidiFile, MidiTrack
from pydantic import ValidationError

from conductor_core.midi import loop_to_midi, midi_to_loop
from conductor_core.models import Loop


def _note_events_with_absolute_times(midi):
    events = []
    absolute_time = 0
    for msg in midi.tracks[0]:
        absolute_time += msg.time
        if msg.type != "end_of_track":
            events.append((msg.type, msg.note, absolute_time))
    return events


def test_midi_helpers_default_to_canonical_numeric_timing(tmp_path, sample_loop):
    midi = MidiFile(ticks_per_beat=480)
    loop_to_midi(midi, sample_loop)
    path = tmp_path / "canonical-default.mid"
    midi.save(path)

    imported = midi_to_loop(str(path))

    assert isinstance(imported, Loop)
    assert imported.Bar_1.notes[0].time.start_beat == 1


def test_loop_to_midi_orders_note_off_before_note_on_at_same_tick(
    loop_factory, note_factory
):
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

    loop_to_midi(midi, loop)

    messages = [msg for msg in midi.tracks[0] if msg.type != "end_of_track"]

    assert [(msg.type, msg.note, msg.time) for msg in messages] == [
        ("note_on", 60, 0),
        ("note_off", 60, 480),
        ("note_on", 64, 0),
        ("note_off", 64, 480),
    ]


def test_loop_to_midi_clamps_out_of_range_velocity(loop_factory, note_factory):
    note = note_factory(pitch="C", start_beat=1, duration=4).model_copy(
        update={"velocity": 200}
    )
    loop = loop_factory(
        bars=[
            [note],
            [],
            [],
            [],
        ]
    )
    midi = MidiFile(ticks_per_beat=480)

    warnings = loop_to_midi(midi, loop)

    note_messages = [msg for msg in midi.tracks[0] if msg.type != "end_of_track"]

    assert [msg.velocity for msg in note_messages] == [127, 127]
    assert warnings == ["Clamped velocity for MIDI note C4 from 200 to 127."]


@pytest.mark.parametrize(
    ("pitch", "octave", "midi_number"),
    [("C", -2, -12), ("C", 12, 156)],
)
def test_loop_to_midi_drops_out_of_range_pitch_pairs(
    loop_factory, note_factory, pitch, octave, midi_number
):
    invalid_note = note_factory(pitch=pitch, start_beat=1, duration=4).model_copy(
        update={"octave": octave}
    )
    loop = loop_factory(
        bars=[
            [note_factory(pitch="G", octave=9, start_beat=1, duration=4)],
            [],
            [],
            [],
        ]
    )
    unvalidated_bar = loop.Bar_1.model_copy(
        update={"notes": [invalid_note, loop.Bar_1.notes[0]]}
    )
    loop = loop.model_copy(update={"Bar_1": unvalidated_bar})
    midi = MidiFile(ticks_per_beat=480)

    warnings = loop_to_midi(midi, loop)

    assert _note_events_with_absolute_times(midi) == [
        ("note_on", 127, 0),
        ("note_off", 127, 480),
    ]
    assert warnings == [
        f"Dropped out-of-range MIDI note {pitch}{octave} ({midi_number}); valid range is 0-127."
    ]


def test_loop_to_midi_drops_non_positive_velocity_pair(loop_factory, note_factory):
    silent_note = note_factory(start_beat=1, duration=4).model_copy(
        update={"velocity": 0}
    )
    loop = loop_factory(bars=[[silent_note], [], [], []])
    midi = MidiFile(ticks_per_beat=480)

    warnings = loop_to_midi(midi, loop)

    assert _note_events_with_absolute_times(midi) == []
    assert warnings == ["Dropped MIDI note C4 with non-positive velocity 0."]


def test_loop_to_midi_drops_non_integer_note_number(loop_factory, note_factory):
    invalid_note = note_factory().model_copy(update={"octave": 4.5})
    loop = loop_factory(bars=[[invalid_note], [], [], []])
    midi = MidiFile(ticks_per_beat=480)

    warnings = loop_to_midi(midi, loop)

    assert _note_events_with_absolute_times(midi) == []
    assert warnings == ["Dropped MIDI note C4.5 with invalid note number 66.0."]


def test_loop_to_midi_allows_notes_to_cross_early_bar_boundaries(
    loop_factory, note_factory
):
    loop = loop_factory(
        bars=[
            [note_factory(pitch="C", start_beat=16, duration=4)],
            [],
            [],
            [],
        ]
    )
    midi = MidiFile(ticks_per_beat=480)

    loop_to_midi(midi, loop)

    assert _note_events_with_absolute_times(midi) == [
        ("note_on", 60, 1800),
        ("note_off", 60, 2280),
    ]


def test_loop_rejects_notes_past_four_bar_boundary(loop_factory, note_factory):
    with pytest.raises(ValidationError, match="four-bar loop boundary"):
        loop_factory(
            bars=[[], [], [], [note_factory(pitch="C", start_beat=16, duration=4)]]
        )


def test_loop_to_midi_preserves_note_at_exact_four_bar_boundary(
    loop_factory, note_factory
):
    loop = loop_factory(
        bars=[[], [], [], [note_factory(pitch="C", start_beat=16, duration=1)]]
    )
    midi = MidiFile(ticks_per_beat=480)

    loop_to_midi(midi, loop)

    assert _note_events_with_absolute_times(midi) == [
        ("note_on", 60, 7560),
        ("note_off", 60, 7680),
    ]


@pytest.mark.parametrize("ticks_per_beat", [0, 1, 2, 3, 5, -24])
def test_loop_to_midi_rejects_ppq_without_an_exact_sixteenth_grid(
    sample_loop, ticks_per_beat
):
    midi = MidiFile(ticks_per_beat=ticks_per_beat)

    with pytest.raises(ValueError, match="positive ticks_per_beat divisible by 4"):
        loop_to_midi(midi, sample_loop)

    assert midi.tracks == []


def test_midi_to_loop_round_trips_integer_timing(sample_loop, midi_builder):
    midi_path = midi_builder(sample_loop)

    loop = midi_to_loop(str(midi_path))

    assert [
        bar.notes[0].pitch for bar in [loop.Bar_1, loop.Bar_2, loop.Bar_3, loop.Bar_4]
    ] == [
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
        bar.notes[0].time.duration == 16
        for bar in [loop.Bar_1, loop.Bar_2, loop.Bar_3, loop.Bar_4]
    )


@pytest.mark.parametrize("note_off_type", ["note_off", "note_on"])
def test_midi_to_loop_ignores_other_channel_note_off(tmp_path, note_off_type):
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack(
        [
            Message("note_on", channel=0, note=60, velocity=96, time=0),
            Message(note_off_type, channel=1, note=60, velocity=0, time=120),
            Message("note_off", channel=0, note=60, velocity=0, time=360),
        ]
    )
    midi.tracks.append(track)
    path = tmp_path / f"cross-channel-{note_off_type}.mid"
    midi.save(path)

    loop = midi_to_loop(str(path))

    assert len(loop.Bar_1.notes) == 1
    assert loop.Bar_1.notes[0].time.duration == 4


@pytest.mark.parametrize("first_channel", [0, 1])
def test_midi_to_loop_pairs_same_pitch_note_offs_by_channel(tmp_path, first_channel):
    velocities = {0: 80, 1: 100}
    second_channel = 1 - first_channel
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack(
        [
            Message("note_on", channel=0, note=60, velocity=velocities[0], time=0),
            Message("note_on", channel=1, note=60, velocity=velocities[1], time=0),
            Message("note_off", channel=first_channel, note=60, velocity=0, time=120),
            Message("note_off", channel=second_channel, note=60, velocity=0, time=360),
        ]
    )
    midi.tracks.append(track)
    path = tmp_path / f"channel-order-{first_channel}.mid"
    midi.save(path)

    loop = midi_to_loop(str(path))

    durations_by_velocity = {
        note.velocity: note.time.duration for note in loop.Bar_1.notes
    }
    assert durations_by_velocity == {
        velocities[first_channel]: 1,
        velocities[second_channel]: 4,
    }


def test_midi_to_loop_pairs_overlapping_same_channel_notes_in_order(tmp_path):
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack(
        [
            Message("note_on", channel=0, note=60, velocity=80, time=0),
            Message("note_on", channel=0, note=60, velocity=100, time=120),
            Message("note_off", channel=0, note=60, velocity=0, time=120),
            Message("note_off", channel=0, note=60, velocity=0, time=240),
        ]
    )
    midi.tracks.append(track)
    path = tmp_path / "same-channel-overlap.mid"
    midi.save(path)

    loop = midi_to_loop(str(path))

    assert [
        (note.velocity, note.time.start_beat, note.time.duration)
        for note in loop.Bar_1.notes
    ] == [(80, 1, 2), (100, 2, 3)]


def test_midi_to_loop_pairs_same_pitch_across_tracks_by_channel(tmp_path):
    midi = MidiFile(ticks_per_beat=480)
    midi.tracks.append(
        MidiTrack(
            [
                Message("note_on", channel=0, note=60, velocity=80, time=0),
                Message("note_off", channel=0, note=60, velocity=0, time=480),
            ]
        )
    )
    midi.tracks.append(
        MidiTrack(
            [
                Message("note_on", channel=1, note=60, velocity=100, time=120),
                Message("note_off", channel=1, note=60, velocity=0, time=240),
            ]
        )
    )
    path = tmp_path / "cross-track-channels.mid"
    midi.save(path)

    loop = midi_to_loop(str(path))

    durations_by_velocity = {
        note.velocity: note.time.duration for note in loop.Bar_1.notes
    }
    assert durations_by_velocity == {80: 4, 100: 2}


@pytest.mark.parametrize("ticks_per_beat", [1, 2, 3, 5])
def test_midi_to_loop_quantizes_low_ppq_timing_to_sixteenths(tmp_path, ticks_per_beat):
    midi = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack(
        [
            Message("note_on", note=60, velocity=96, time=4 * ticks_per_beat),
            Message("note_off", note=60, velocity=0, time=ticks_per_beat),
        ]
    )
    midi.tracks.append(track)
    path = tmp_path / f"ppq-{ticks_per_beat}.mid"
    midi.save(path)

    loop = midi_to_loop(str(path))

    assert loop.Bar_1.notes == []
    assert len(loop.Bar_2.notes) == 1
    assert loop.Bar_2.notes[0].time.start_beat == 1
    assert loop.Bar_2.notes[0].time.duration == 4


@pytest.mark.parametrize("ticks_per_beat", [0, -24])
def test_midi_to_loop_rejects_non_ppq_time_divisions(tmp_path, ticks_per_beat):
    midi = MidiFile(ticks_per_beat=ticks_per_beat)
    midi.tracks.append(MidiTrack())
    path = tmp_path / f"division-{ticks_per_beat}.mid"
    midi.save(path)

    with pytest.raises(ValueError, match="positive PPQ time division"):
        midi_to_loop(str(path))


def test_midi_to_loop_skips_notes_beyond_the_first_four_bars(tmp_path):
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    track.append(Message("note_on", note=60, velocity=96, time=7680))
    track.append(Message("note_off", note=60, velocity=96, time=120))
    track.append(MetaMessage("end_of_track", time=0))
    midi.tracks.append(track)

    midi_path = tmp_path / "fifth_bar.mid"
    midi.save(midi_path)

    loop = midi_to_loop(str(midi_path))

    assert loop.Bar_1.notes == []
    assert loop.Bar_2.notes == []
    assert loop.Bar_3.notes == []
    assert loop.Bar_4.notes == []


def test_midi_to_loop_imports_seventeen_sixteenth_sustains(tmp_path):
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

    loop = midi_to_loop(str(path))

    duration = loop.Bar_1.notes[0].time.duration
    assert duration == 17


def test_midi_long_note_round_trips_across_multiple_bars(tmp_path):
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

    loop = midi_to_loop(str(source))
    exported = MidiFile(ticks_per_beat=480)
    loop_to_midi(exported, loop)

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

    loop = midi_to_loop(str(path))

    assert loop.Bar_4.notes[0].time.start_beat == 16
    assert loop.Bar_4.notes[0].time.duration == 1
