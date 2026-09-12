"""Smoke-test the built wheel from a consumer-style installation."""

from importlib import resources

import conductor_core
from conductor_core import (
    EngineConfig,
    GenerationMetadata,
    GenerationRequest,
    GenerationResult,
    LoopGenerationEngine,
    ProgressEvent,
    ProviderCredentials,
    VariationBatchMetadata,
    VariationBatchResult,
    VariationDiagnostic,
    VariationResult,
    VariationUsage,
    resolve_conductor_home,
    resolve_data_dir,
    resolve_default_artifact_root,
    validate_variation_count,
)
from conductor_core.music import (
    VARIATION_PROMPT_VERSION,
    get_loop_prompt,
    get_model_info,
    get_variation_prompt,
)

expected_public_api = {
    "EngineConfig",
    "GenerationMetadata",
    "GenerationRequest",
    "GenerationResult",
    "LoopGenerationEngine",
    "ProgressEvent",
    "ProviderCredentials",
    "VariationBatchMetadata",
    "VariationBatchResult",
    "VariationDiagnostic",
    "VariationResult",
    "VariationUsage",
    "resolve_conductor_home",
    "resolve_data_dir",
    "resolve_default_artifact_root",
    "validate_variation_count",
}
assert expected_public_api <= set(conductor_core.__all__)

public_imports = (
    EngineConfig,
    GenerationMetadata,
    GenerationRequest,
    GenerationResult,
    LoopGenerationEngine,
    ProgressEvent,
    ProviderCredentials,
    VariationBatchMetadata,
    VariationBatchResult,
    VariationDiagnostic,
    VariationResult,
    VariationUsage,
    resolve_conductor_home,
    resolve_data_dir,
    resolve_default_artifact_root,
    validate_variation_count,
)
assert all(public_import is not None for public_import in public_imports)

py_typed = resources.files("conductor_core").joinpath("py.typed")
assert py_typed.is_file()

model_info = get_model_info()
assert isinstance(model_info.get("models"), dict)
assert model_info["models"]

loop_prompt = get_loop_prompt()
assert isinstance(loop_prompt, str)
assert loop_prompt.strip()

assert VARIATION_PROMPT_VERSION == "variation_gen_v1"
variation_prompt = get_variation_prompt()
assert isinstance(variation_prompt, str)
assert variation_prompt.strip()

soundfont = resources.files("conductor_core.resources").joinpath(
    "soundfonts",
    "FM-Piano1-20190916.sf2",
)
assert soundfont.is_file()
with soundfont.open("rb") as soundfont_file:
    assert soundfont_file.read(4) == b"RIFF"
