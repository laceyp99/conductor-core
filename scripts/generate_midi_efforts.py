"""Generate one MIDI loop per advertised reasoning effort for a model.

Running this script starts one paid generation per effort. Each generation
uses the same musical prompt and model pricing. Token usage and actual cost
can vary by effort. The generation API has no per-request dollar cap.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from generate_midi import (
    DESCRIPTION,
    KEY,
    PROMPT_OVERRIDE,
    SCALE,
    SOUNDFONT_PATH,
    TEMPERATURE,
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
        # so workers cannot race to delete the same older generation. A later
        # normal generation can prune the history back to its configured limit.
        config = EngineConfig.from_defaults(max_generations=None)
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
            use_thinking=True,
            effort=effort,
            prompt_override=PROMPT_OVERRIDE,
            render_audio=args.render_audio,
            soundfont_path=SOUNDFONT_PATH,
        )
        return engine.generate(request)

    results = {}
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        try:
            pending = {executor.submit(generate, effort): effort for effort in efforts}
            unfinished = set(pending)
            while unfinished:
                # A bounded wait lets Windows deliver Ctrl+C while workers run.
                done, unfinished = wait(
                    unfinished, timeout=0.5, return_when=FIRST_COMPLETED
                )
                for future in done:
                    effort = pending[future]
                    prefix = f"[{effort}]"
                    if (exc := future.exception()) is not None:
                        failed += 1
                        print(f"{prefix} Failed: {exc}")
                        continue
                    result = future.result()
                    results[effort] = result
                    print(f"{prefix} MIDI: {result.midi_path}")
                    if result.audio_path is not None:
                        print(f"{prefix} Audio: {result.audio_path}")
                    cost = (
                        f"${result.cost:.6f}"
                        if result.cost is not None
                        else "unavailable"
                    )
                    print(f"{prefix} Estimated provider cost: {cost}")
                    for warning in result.warnings:
                        print(f"{prefix} Warning: {warning}")
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            print(
                "Interrupted. Waiting for generations already in progress.", flush=True
            )
            raise

    if failed:
        parser.exit(1, f"{failed} of {len(efforts)} generations failed.\n")
    return {effort: results[effort] for effort in efforts}


if __name__ == "__main__":
    main()
