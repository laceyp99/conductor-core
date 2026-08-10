"""Smoke-test the built wheel from a consumer-style installation."""

from importlib import resources

import conductor_core
from conductor_core import (
    EngineConfig,
    GenerationRequest,
    GenerationResult,
    LoopGenerationEngine,
    ProgressEvent,
    ProviderCredentials,
    resolve_conductor_home,
    resolve_data_dir,
    resolve_default_artifact_root,
)
from conductor_core.music import get_loop_prompt, get_model_info

expected_public_api = {
    "EngineConfig",
    "GenerationRequest",
    "GenerationResult",
    "LoopGenerationEngine",
    "ProgressEvent",
    "ProviderCredentials",
    "resolve_conductor_home",
    "resolve_data_dir",
    "resolve_default_artifact_root",
}
assert expected_public_api <= set(conductor_core.__all__)

public_imports = (
    EngineConfig,
    GenerationRequest,
    GenerationResult,
    LoopGenerationEngine,
    ProgressEvent,
    ProviderCredentials,
    resolve_conductor_home,
    resolve_data_dir,
    resolve_default_artifact_root,
)
assert all(public_import is not None for public_import in public_imports)

model_info = get_model_info()
assert isinstance(model_info.get("models"), dict)
assert model_info["models"]

loop_prompt = get_loop_prompt()
assert isinstance(loop_prompt, str)
assert loop_prompt.strip()

soundfont = (
    resources.files("conductor_core.resources")
    .joinpath("soundfonts")
    .joinpath("FM-Piano1-20190916.sf2")
)
assert soundfont.is_file()
with soundfont.open("rb") as soundfont_file:
    assert soundfont_file.read(4) == b"RIFF"
