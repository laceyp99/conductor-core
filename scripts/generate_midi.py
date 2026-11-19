"""Generate a MIDI loop with a provider and optionally render its audio.

WARNING: Running this file makes a real provider request. It requires the provider's
SDK and credentials, may send the musical description to that provider, and may
incur usage charges. Audio rendering additionally requires FluidSynth and FFmpeg
with MP3 encoding support on PATH.

Run ``python scripts/generate_midi.py MODEL --effort high`` from the repository
root. The model defaults to the value below, and reasoning effort defaults
to low. Importing this module does not make a provider call.
"""

import argparse

from conductor_core import EngineConfig, GenerationRequest, LoopGenerationEngine

# --- Edit these values before running the example. ---
MODEL = "gemini-4-hydrogen"
KEY = "A"
SCALE = "minor"
DESCRIPTION = (
    "warm neo-soul electric piano chords with syncopated upper extensions "
    "and a simple bass movement"
)
TEMPERATURE = 0.0
EFFORT = "low"
RENDER_AUDIO = False

# None uses Conductor Core's packaged default prompt. To customize the system
# prompt for this request, replace None with text such as the commented example.
PROMPT_OVERRIDE = None
# PROMPT_OVERRIDE = "Generate sparse piano-only material using the required schema."

# None asks the playback helpers to resolve Core's default packaged SoundFont.
SOUNDFONT_PATH = None


def report_progress(event):
    """Print progress from the synchronous generation engine."""
    detail = f" ({event.detail})" if event.detail else ""
    print(f"[{event.stage}] {event.message}{detail}")


def add_render_audio_argument(parser, *, default=RENDER_AUDIO):
    """Add the audio option shared by the generation examples."""

    def parse_boolean(value):
        value = value.lower()
        if value not in ("true", "false"):
            raise argparse.ArgumentTypeError("expected true or false")
        return value == "true"

    parser.add_argument(
        "--render-audio",
        type=parse_boolean,
        default=default,
        metavar="{true,false}",
        help=f"Render MP3s (default: {str(default).lower()})",
    )


def main(engine=None, *, model=MODEL, effort=EFFORT, render_audio=RENDER_AUDIO):
    """Run the configured generation and return its result."""
    if engine is None:
        config = EngineConfig.from_defaults()
        engine = LoopGenerationEngine(config)

    request = GenerationRequest(
        key=KEY,
        scale=SCALE,
        description=DESCRIPTION,
        model=model,
        temperature=TEMPERATURE,
        use_thinking=effort != "off",
        effort=None if effort == "off" else effort,
        prompt_override=PROMPT_OVERRIDE,
        render_audio=render_audio,
        soundfont_path=SOUNDFONT_PATH,
    )
    result = engine.generate(request, progress_callback=report_progress)

    print("\nGeneration complete")
    print(f"Effort: {effort}")
    print(f"ID: {result.generation_id}")
    print(f"MIDI: {result.midi_path}")
    print(f"Audio: {result.audio_path or 'not rendered'}")
    print(
        f"Estimated provider cost: {result.cost if result.cost is not None else 'unavailable'}"
    )
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default=MODEL, help="Provider model ID")
    parser.add_argument(
        "--effort",
        default=EFFORT,
        help="Model effort level, or off to turn thinking off (default: %(default)s)",
    )
    add_render_audio_argument(parser)
    main(**vars(parser.parse_args()))
