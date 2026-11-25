"""
Unit tests for Stage 5: Output Generation & Serialization.
"""

import json
from config.settings import PipelineStage, StageStatus
from pipeline.output_handler import OutputHandler, StageRecord


def test_output_handler_bundle_generation():
    handler = OutputHandler()
    stage_records = {
        PipelineStage.INPUT_PROCESSING.value: StageRecord(
            stage=PipelineStage.INPUT_PROCESSING,
            status=StageStatus.COMPLETED,
            latency_seconds=0.01,
        )
    }

    bundle = handler.generate_bundle(
        original_input="Test prompt",
        stage_records=stage_records,
        model_used="gpt-4o-mini",
        style_applied="Concise & Direct",
        total_time=1.234,
    )

    assert bundle.original_input == "Test prompt"
    assert bundle.model_used == "gpt-4o-mini"
    assert bundle.total_execution_time_seconds == 1.234


def test_output_handler_format_as_text():
    handler = OutputHandler()
    stage_records = {
        PipelineStage.INPUT_PROCESSING.value: StageRecord(
            stage=PipelineStage.INPUT_PROCESSING,
            status=StageStatus.COMPLETED,
            latency_seconds=0.05,
        )
    }
    bundle = handler.generate_bundle(
        original_input="Test question",
        stage_records=stage_records,
        total_time=0.5,
    )

    text_report = handler.format_as_text(bundle)
    assert "LLM PIPELINE EXECUTION REPORT" in text_report
    assert "ORIGINAL USER INPUT:" in text_report
    assert "FINAL GENERATED OUTPUT:" in text_report


def test_output_handler_format_as_json():
    handler = OutputHandler()
    bundle = handler.generate_bundle(
        original_input="Sample prompt",
        stage_records={},
        total_time=0.2,
    )

    json_str = handler.format_as_json(bundle)
    data = json.loads(json_str)
    assert "timestamp" in data
    assert "total_execution_time_seconds" in data
    assert data["model_used"] == "gpt-4o-mini"
