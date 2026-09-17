from importlib import resources

from conductor_core.music import VARIATION_PROMPT_VERSION, get_variation_prompt


def test_variation_prompt_version_identifies_packaged_resource():
    assert VARIATION_PROMPT_VERSION == "variation_gen_v1"

    prompt_resource = resources.files("conductor_core.resources").joinpath(
        "prompts", f"{VARIATION_PROMPT_VERSION}.txt"
    )
    assert prompt_resource.is_file()


def test_get_variation_prompt_loads_standalone_batch_instructions():
    prompt = get_variation_prompt()

    assert prompt.strip()
    assert 'only top-level property is "items"' in prompt
    assert 'value of "items" must be an array' in prompt
    assert "additional top-level properties" in prompt
    assert "exactly the requested number" in prompt
    assert "2 through 8" in prompt
    assert "coherent but distinct alternatives" in prompt
    assert "Bar_1" in prompt
    assert "Bar_4" in prompt
    assert "start_beat" in prompt
    assert "duration" in prompt
    assert "beyond the end of bar 4" in prompt
    assert "Markdown" in prompt
