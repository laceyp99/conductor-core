"""Generate several MIDI loop alternatives with one provider request.

WARNING: Running this file makes a real provider request. It requires the provider's
SDK and credentials, may send the musical description to that provider, and may
incur usage charges. Audio rendering additionally requires the ``playback`` extra,
FluidSynth, and FFmpeg.

Edit the constants below, then run ``python scripts/generate_variations.py`` from
the repository root. Importing this module does not make a provider call.
"""

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
USE_THINKING = True
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


def main(engine=None):
    """Run the configured batch generation and return its result."""
    if engine is None:
        config = EngineConfig.from_defaults()
        engine = LoopGenerationEngine(config)

    request = VariationGenerationRequest(
        key=KEY,
        scale=SCALE,
        description=DESCRIPTION,
        model=MODEL,
        count=COUNT,
        temperature=TEMPERATURE,
        use_thinking=USE_THINKING,
        effort=EFFORT,
        prompt_override=PROMPT_OVERRIDE,
        render_audio=RENDER_AUDIO,
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
    main()
