"""Convert an existing MIDI file to a quantized Core loop and back to MIDI.

This example is entirely offline. Conductor Core reads notes from the first four
bars, maps starts and durations to sixteenth-note integer positions, and writes a
new normalized MIDI file. Edit the two paths below, then run this file from the
repository root.
"""

from pathlib import Path

from mido import MidiFile

from conductor_core import resolve_default_artifact_root
from conductor_core.midi import loop_to_midi, midi_to_loop

# --- Edit these paths before running the example. ---
INPUT_MIDI_PATH = resolve_default_artifact_root() / "gen_<id>" / "loop.mid"
OUTPUT_MIDI_PATH = resolve_default_artifact_root() / "gen_<id>" / "roundtrip.mid"


def roundtrip_midi(input_path, output_path):
    """Quantize a MIDI file to Core's sixteenth-note loop and write a new MIDI."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.resolve() == output_path.resolve():
        raise ValueError("INPUT_MIDI_PATH and OUTPUT_MIDI_PATH must be different files.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input MIDI file not found: {input_path}")

    loop = midi_to_loop(str(input_path), times_as_string=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi = MidiFile()
    loop_to_midi(midi, loop, times_as_string=False)
    midi.save(output_path)
    return loop


def note_count(loop):
    """Return the number of notes across the loop's four bars."""
    return sum(len(getattr(loop, f"Bar_{number}").notes) for number in range(1, 5))


def main():
    """Run the configured offline round-trip."""
    loop = roundtrip_midi(INPUT_MIDI_PATH, OUTPUT_MIDI_PATH)
    print(f"Read: {INPUT_MIDI_PATH}")
    print(f"Quantized notes in first four bars: {note_count(loop)}")
    print(f"Wrote: {OUTPUT_MIDI_PATH}")
    return loop


if __name__ == "__main__":
    main()
