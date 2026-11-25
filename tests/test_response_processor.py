"""
Unit tests for Stage 4: Response Processing.
"""

from pipeline.response_processor import ResponseProcessor


def test_clean_response_normalizes_whitespace():
    processor = ResponseProcessor()
    raw = "Here is the response.   \n\n\n\nNext paragraph.  "
    cleaned = processor.clean_response(raw)
    assert cleaned == "Here is the response.\n\nNext paragraph."


def test_count_sentences():
    processor = ResponseProcessor()
    text = "First sentence. Second sentence! Third sentence? Done."
    count = processor.count_sentences(text)
    assert count == 4


def test_count_code_blocks():
    processor = ResponseProcessor()
    text = (
        "Here is some python code:\n"
        "```python\nprint('hello')\n```\n"
        "And another block:\n"
        "```bash\necho 123\n```"
    )
    assert processor.count_code_blocks(text) == 2


def test_extract_key_points():
    processor = ResponseProcessor()
    text = (
        "Key aspects:\n"
        "- First major finding\n"
        "- Second major finding\n"
        "1. Third numbered item\n"
        "Conclusion text."
    )
    points = processor.extract_key_points(text)
    assert len(points) == 3
    assert points[0] == "First major finding"
    assert points[1] == "Second major finding"
    assert points[2] == "Third numbered item"


def test_process_empty_response():
    processor = ResponseProcessor()
    res = processor.process(raw_response="")
    assert res.is_valid is False
    assert "empty" in res.error_message.lower()


def test_process_valid_response():
    processor = ResponseProcessor()
    raw = (
        "Quantum computing harnesses superposition and entanglement.\n"
        "- Superposition allows qubits to exist in multiple states simultaneously.\n"
        "- Entanglement correlates qubit pairs instantly.\n"
        "This offers exponential speedup for specific computational problems."
    )
    res = processor.process(raw_response=raw, completion_tokens=35, latency_seconds=1.5)

    assert res.is_valid is True
    assert res.word_count > 0
    assert res.sentence_count >= 3
    assert len(res.key_points) == 2
    assert res.tokens_per_second > 0
    assert res.estimated_reading_time_minutes > 0
