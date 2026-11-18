"""Generate several MIDI loop alternatives with one provider request.

WARNING: Running this file makes a real provider request. It requires the provider's
SDK and credentials, may send the musical description to that provider, and may
incur usage charges. Audio rendering additionally requires FluidSynth and FFmpeg
with MP3 encoding support on PATH.

Run ``python scripts/generate_variations.py MODEL --effort high --count 4``
from the repository root. Importing this module does not make a provider call.
"""

import argparse

from conductor_core import (
    EngineConfig,
    LoopGenerationEngine,
    VariationGenerationRequest,
)

# --- Edit these values before running the example. ---
MODEL = "gpt-5.6-luna"
KEY = "A"
SCALE = "minor"
DESCRIPTION = (
    "warm neo-soul electric piano chords with syncopated upper extensions "
    "and a simple bass movement"
)
COUNT = 4
TEMPERATURE = 0.0
EFFORT = "medium"
RENDER_AUDIO = True

# None uses Conductor Core's packaged variation prompt. To customize the complete
# system prompt for this request, replace None with text such as this example:
PROMPT_OVERRIDE = None
# PROMPT_OVERRIDE = "Generate distinct piano-only alternatives using the schema."

# None asks the playback helpers to resolve Core's default packaged SoundFont.
SOUNDFONT_PATH = None


def report_progress(event):
    """Print correlated progress from the synchronous variation engine."""
    item = (
        f" variation={event.variation_index + 1}"
        if event.variation_index is not None
        else ""
    )
    status = f" status={event.status}" if event.status else ""
    detail = f" ({event.detail})" if event.detail else ""
    print(f"[{event.stage}]{item}{status} {event.message}{detail}")


def main(
    engine=None, *, model=MODEL, effort=EFFORT, count=COUNT, render_audio=RENDER_AUDIO
):
    """Run the configured batch generation and return its result."""
    if engine is None:
        config = EngineConfig.from_defaults()
        engine = LoopGenerationEngine(config)

    request = VariationGenerationRequest(
        key=KEY,
        scale=SCALE,
        description=DESCRIPTION,
        model=model,
        count=count,
        temperature=TEMPERATURE,
        use_thinking=effort != "off",
        effort=None if effort == "off" else effort,
        prompt_override=PROMPT_OVERRIDE,
        render_audio=render_audio,
        soundfont_path=SOUNDFONT_PATH,
    )
    result = engine.generate_variations(request, progress_callback=report_progress)

    print(f"\nVariation batch {result.status}")
    print(f"Batch ID: {result.metadata.batch_id}")
    print(
        "Estimated provider cost: "
        f"{result.metadata.cost if result.metadata.cost is not None else 'unavailable'}"
    )
    if result.diagnostic is not None:
        print(f"Diagnostic: {result.diagnostic.code}: {result.diagnostic.message}")

    for item in result.items:
        print(f"\nVariation {item.index + 1}")
        print(f"ID: {item.generation.id}")
        print(f"MIDI: {item.generation.midi_path}")
        print(f"Audio: {item.generation.audio_path or 'not rendered'}")
        if item.warnings:
            print("Warnings:")
            for warning in item.warnings:
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
    parser.add_argument(
        "--count",
        type=int,
        default=COUNT,
        help="Number of variations, from 2 to 8 (default: %(default)s)",
    )
    parser.add_argument(
        "--render-audio",
        action=argparse.BooleanOptionalAction,
        default=RENDER_AUDIO,
        help="Render MP3s (enabled by default)",
    )
    main(**vars(parser.parse_args()))
