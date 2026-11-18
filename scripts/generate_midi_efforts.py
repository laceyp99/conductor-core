"""Generate one MIDI loop per advertised reasoning effort for a model.

Running this script makes one paid provider request per effort. Each request
uses the same musical prompt and model pricing. Token usage and actual cost
can vary by effort. The generation API has no per-request dollar cap.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

from generate_midi import (
    DESCRIPTION,
    KEY,
    PROMPT_OVERRIDE,
    SCALE,
    SOUNDFONT_PATH,
    TEMPERATURE,
    provider_timing,
)

from conductor_core import EngineConfig, GenerationRequest, LoopGenerationEngine
from conductor_core.music import get_model_info


def main(argv=None, *, engine=None):
    """Run the same prompt once at each configured effort."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="A model ID from the packaged registry")
    parser.add_argument("--render-audio", action="store_true", help="Also render MP3s")
    parser.add_argument("--workers", type=int, help="Limit concurrent requests")
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")

    model_config = next(
        (
            models[args.model]
            for models in get_model_info()["models"].values()
            if args.model in models
        ),
        None,
    )
    if model_config is None:
        parser.error(f"Unknown model: {args.model}")
    efforts = model_config.get("effort_options", [])
    if not efforts:
        parser.error(f"{args.model} has no configurable effort levels")

    if engine is None:
        # Concurrent generations use separate workspaces. Skip history pruning
        # so workers cannot race to delete the same older generation.
        config = replace(EngineConfig.from_defaults(), max_generations=None)
        engine = LoopGenerationEngine(config)

    workers = min(args.workers or len(efforts), len(efforts))
    print(f"{args.model}: {len(efforts)} requests, up to {workers} at once")

    def generate(effort):
        request = GenerationRequest(
            key=KEY,
            scale=SCALE,
            description=DESCRIPTION,
            model=args.model,
            temperature=TEMPERATURE,
            use_thinking=effort != "none",
            effort=effort,
            prompt_override=PROMPT_OVERRIDE,
            render_audio=args.render_audio,
            soundfont_path=SOUNDFONT_PATH,
        )
        return engine.generate(
            request, progress_callback=provider_timing(f"[{effort}] ")
        )

    results = {}
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(generate, effort): effort for effort in efforts}
        for future in as_completed(pending):
            effort = pending[future]
            if (exc := future.exception()) is not None:
                failed += 1
                print(f"\n[{effort}] Failed: {exc}")
                continue
            result = future.result()
            results[effort] = result
            print(f"\n[{effort}] MIDI: {result.midi_path}")
            if result.audio_path is not None:
                print(f"Audio: {result.audio_path}")
            cost = f"${result.cost:.6f}" if result.cost is not None else "unavailable"
            print(f"Estimated provider cost: {cost}")
            for warning in result.warnings:
                print(f"Warning: {warning}")

    if failed:
        parser.exit(1, f"{failed} of {len(efforts)} generations failed.\n")
    return {effort: results[effort] for effort in efforts}


if __name__ == "__main__":
    main()
