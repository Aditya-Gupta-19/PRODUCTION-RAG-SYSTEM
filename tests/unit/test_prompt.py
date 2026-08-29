import pytest

from src.generation.prompt import Prompt, load_prompt


def test_loads_rag_v1_from_yaml():
    prompt = load_prompt("rag_v1")
    assert isinstance(prompt, Prompt)
    assert prompt.version == "rag_v1"
    assert "ONLY" in prompt.system
    assert "{context}" in prompt.user_template
    assert "{question}" in prompt.user_template


def test_renders_user_message_with_substitutions():
    prompt = load_prompt("rag_v1")
    rendered = prompt.render_user(context="[1] (a.txt, page 3)\nsome fact", question="what is the fact?")
    assert "[1] (a.txt, page 3)" in rendered
    assert "what is the fact?" in rendered
    assert "{context}" not in rendered


def test_unknown_version_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist_v9")


def test_load_prompt_is_cached():
    assert load_prompt("rag_v1") is load_prompt("rag_v1")


def test_model_defaults_exposed():
    assert load_prompt("rag_v1").model.get("temperature") == 0.1
