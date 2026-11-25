"""
Unit tests for Stage 2: Prompt Engineering.
"""

from config.settings import OUTPUT_STYLES
from pipeline.prompt_builder import PERSONAS, PromptBuilder


def test_prompt_builder_system_prompt_includes_style_and_persona():
    builder = PromptBuilder()
    sys_prompt = builder.build_system_prompt(
        style_name="Concise & Direct",
        persona_name="Senior Software Engineer",
    )

    assert PERSONAS["Senior Software Engineer"] in sys_prompt
    assert OUTPUT_STYLES["Concise & Direct"]["instruction"] in sys_prompt
    assert "Core Guidelines:" in sys_prompt


def test_prompt_builder_includes_custom_instructions():
    builder = PromptBuilder()
    custom_inst = "Never use the letter 'z' in your answer."
    sys_prompt = builder.build_system_prompt(
        custom_system_instruction=custom_inst,
    )
    assert custom_inst in sys_prompt


def test_prompt_builder_user_prompt_formatting():
    builder = PromptBuilder()
    user_prompt = builder.build_user_prompt("What is Docker?")
    assert "```\nWhat is Docker?\n```" in user_prompt


def test_prompt_builder_with_task_context():
    builder = PromptBuilder()
    user_prompt = builder.build_user_prompt(
        cleaned_input="def foo(): pass",
        task_context="Optimize for memory",
    )
    assert "Context / Task Goal:\nOptimize for memory" in user_prompt
    assert "def foo(): pass" in user_prompt


def test_prompt_builder_complete_build():
    builder = PromptBuilder()
    result = builder.build(
        cleaned_input="Summarize this text.",
        style_name="Bullet Points & Summary",
        persona_name="Executive Product Manager",
    )

    assert len(result.messages) == 2
    assert result.messages[0]["role"] == "system"
    assert result.messages[1]["role"] == "user"
    assert result.style_applied == "Bullet Points & Summary"
    assert result.persona_applied == "Executive Product Manager"
    assert result.total_prompt_length > 0
    assert result.estimated_prompt_tokens > 0
