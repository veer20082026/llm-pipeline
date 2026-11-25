"""
Unit tests for Stage 1: Input Processing.
"""

import pytest
from pipeline.input_processor import InputProcessor


def test_clean_text_normalizes_spaces_and_newlines():
    processor = InputProcessor()
    dirty_text = "   Hello    world!   \n\n\n\nThis  is    a   test.   \r\n"
    cleaned = processor.clean_text(dirty_text)
    assert cleaned == "Hello world!\n\nThis is a test."


def test_clean_text_removes_unprintable_control_characters():
    processor = InputProcessor()
    text_with_null = "Hello\x00World\x07!"
    cleaned = processor.clean_text(text_with_null)
    assert cleaned == "HelloWorld!"


def test_validation_empty_input():
    processor = InputProcessor(min_length=3)
    is_valid, error = processor.validate("")
    assert not is_valid
    assert "empty" in error.lower()


def test_validation_whitespace_only():
    processor = InputProcessor(min_length=3)
    is_valid, error = processor.validate("     \n\t   ")
    assert not is_valid
    assert "empty" in error.lower()


def test_validation_too_short():
    processor = InputProcessor(min_length=10)
    is_valid, error = processor.validate("Hi there")
    assert not is_valid
    assert "too short" in error.lower()


def test_validation_too_long():
    processor = InputProcessor(max_length=20)
    is_valid, error = processor.validate("This string is definitely longer than twenty characters.")
    assert not is_valid
    assert "exceeds maximum length" in error.lower()


def test_process_success():
    processor = InputProcessor(min_length=5, max_length=500)
    raw = "   Explain   neural   networks   in   simple   terms.   "
    result = processor.process(raw)

    assert result.is_valid is True
    assert result.cleaned_text == "Explain neural networks in simple terms."
    assert result.word_count == 6
    assert result.character_count == len("Explain neural networks in simple terms.")
    assert result.estimated_tokens > 0
    assert result.validation_error is None


def test_process_failure_records_error():
    processor = InputProcessor(min_length=10)
    result = processor.process("Hi")
    assert result.is_valid is False
    assert result.validation_error is not None
    assert result.word_count == 0
