import os
import subprocess
import sys
from pathlib import Path


def test_core_import_does_not_load_ui_dashboard_or_eval_modules():
    core_src = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(core_src)
    code = (
        "import conductor_core, sys; "
        "blocked = ['gradio', 'dash', 'plotly', 'evaluation']; "
        "print({name: name in sys.modules for name in blocked})"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == (
        "{'gradio': False, 'dash': False, 'plotly': False, 'evaluation': False}"
    )


def test_routing_imports_without_optional_provider_sdks():
    core_src = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(core_src)
    code = r"""
import importlib.abc
import sys

blocked = {"anthropic", "google", "httpx", "ollama", "openai"}

class BlockOptionalProviders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise ImportError(f"blocked optional SDK: {fullname}")
        return None

sys.meta_path.insert(0, BlockOptionalProviders())
import conductor_core.routing
from conductor_core._internal_types import ProviderVariationResult
print(ProviderVariationResult.__name__)
print([name for name in blocked if name in sys.modules])
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.splitlines() == ["ProviderVariationResult", "[]"]
