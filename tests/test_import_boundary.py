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


def test_missing_provider_sdks_keep_imports_safe_and_generation_errors_actionable():
    core_src = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(core_src)
    code = """
import importlib.abc
import sys

class BlockProviderSDKs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'anthropic', 'google', 'httpx', 'ollama', 'openai'}:
            raise ImportError('SDK blocked for minimal-install test')

sys.meta_path.insert(0, BlockProviderSDKs())
import conductor_core
from conductor_core.providers import anthropic, google, ollama, openai
for provider, extra in ((anthropic, 'anthropic'), (google, 'google'),
                        (ollama, 'ollama'), (openai, 'openai')):
    for generate in (provider.loop_gen, provider.variations_gen):
        try:
            generate(prompt='melody', model='unavailable')
        except ImportError as exc:
            assert f'conductor-core[{extra}]' in str(exc), str(exc)
        else:
            raise AssertionError(f'{extra} did not report the missing SDK')
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
