"""Generate a MIDI loop with a provider and optionally render its audio.

WARNING: Running this file makes a real provider request. It requires the provider's
SDK and credentials, may send the musical description to that provider, and may
incur usage charges. Audio rendering additionally requires the ``playback`` extra,
FluidSynth, and FFmpeg.

Edit the constants below, then run ``python scripts/generate_midi.py`` from the
repository root. Importing this module does not make a provider call.
"""

from pathlib import Path

from conductor_core import EngineConfig, GenerationRequest, LoopGenerationEngine

# --- Edit these values before running the example. ---
MODEL = "gpt-5.6-terra"
KEY = "A"
SCALE = "minor"
DESCRIPTION = (
    "warm neo-soul electric piano chords with syncopated upper extensions "
    "and a simple bass movement"
)
TEMPERATURE = 0.0
USE_THINKING = True
EFFORT = "medium"
ARTIFACT_ROOT = Path("generations")
RENDER_AUDIO = True

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


def main(engine=None):
    """Run the configured generation and return its result."""
    if engine is None:
        config = EngineConfig.from_defaults(artifact_root=ARTIFACT_ROOT)
        engine = LoopGenerationEngine(config)

    request = GenerationRequest(
        key=KEY,
        scale=SCALE,
        description=DESCRIPTION,
        model=MODEL,
        temperature=TEMPERATURE,
        use_thinking=USE_THINKING,
        effort=EFFORT,
        prompt_override=PROMPT_OVERRIDE,
        render_audio=RENDER_AUDIO,
        soundfont_path=SOUNDFONT_PATH,
    )
    result = engine.generate(request, progress_callback=report_progress)

    print("\nGeneration complete")
    print(f"ID: {result.generation_id}")
    print(f"MIDI: {result.midi_path}")
    print(f"Audio: {result.audio_path or 'not rendered'}")
    print(f"Estimated provider cost: {result.cost if result.cost is not None else 'unavailable'}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")

    return result


if __name__ == "__main__":
    main()
