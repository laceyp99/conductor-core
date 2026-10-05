"""Static consumer contract sample; checked against an installed wheel only."""

from importlib import resources
from pathlib import Path

from typing_extensions import assert_type

from conductor_core import (
    EngineConfig,
    GenerationRequest,
    GenerationResult,
    LoopGenerationEngine,
    ProgressCallback,
    ProgressEvent,
    VariationBatchResult,
    VariationGenerationRequest,
    VariationHistoryListResult,
    VariationHistoryLookupResult,
    list_variation_history,
)
from conductor_core.models import Loop
from conductor_core.music import get_loop_prompt, get_model_info
from conductor_core.routing import LoopProviderResult, VariationProviderResult
from conductor_core.storage import FilesystemArtifactStore, GenerationMetadata


def progress(event: ProgressEvent) -> str:
    assert_type(event.message, str)
    return event.message


def consume(root: Path) -> None:
    callback: ProgressCallback = progress
    store = FilesystemArtifactStore(root, max_generations=None)
    engine = LoopGenerationEngine(EngineConfig(artifact_root=root), store=store)
    request = GenerationRequest("C", "major", "A melody", "gpt-6-sol")
    result = engine.generate(request, progress_callback=callback)
    assert_type(result, GenerationResult)
    assert_type(result.loop, Loop)
    assert_type(result.messages[0]["content"], str)
    assert_type(result.metadata, GenerationMetadata)
    assert_type(result.audio_path, str | None)

    variations = engine.generate_variations(
        VariationGenerationRequest("C", "major", "Alternatives", "gpt-6-sol"),
        progress_callback=callback,
    )
    assert_type(variations, VariationBatchResult)
    assert_type(variations.items[0].loop, Loop)
    assert_type(variations.metadata.messages[0]["content"], str)
    assert_type(
        store.get_variation_history(variations.metadata.batch_id),
        VariationHistoryLookupResult,
    )
    assert_type(store.list_variation_history(limit=None), VariationHistoryListResult)
    assert_type(store.list_variation_history(limit=None).limit, int | None)
    workspace = store.create_generation_workspace()
    metadata = store.finalize_generation(
        workspace, "A melody", "C", "major", "gpt-6-sol", "OpenAI", 0.0
    )
    assert_type(metadata, GenerationMetadata)


def consume_routing(single: LoopProviderResult, batch: VariationProviderResult) -> None:
    assert_type(single.loop, Loop)
    assert_type(single.messages[0]["content"], str)
    assert_type(batch.variations, tuple[Loop, ...])


assert resources.files("conductor_core").joinpath("py.typed").is_file()
assert_type(get_loop_prompt(), str)
assert_type(get_model_info()["models"]["OpenAI"]["gpt-6-sol"]["max_tokens"], int)
assert_type(get_model_info()["models"]["OpenAI"]["gpt-6-sol"]["cost"]["input"], float)
assert_type(list_variation_history(limit=None), VariationHistoryListResult)
assert_type(list_variation_history(limit=None).limit, int | None)
