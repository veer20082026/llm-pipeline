"""
Integration tests for the complete end-to-end Pipeline Runner.
"""

from unittest.mock import MagicMock, patch
from config.settings import StageStatus
from pipeline.pipeline_runner import PipelineRunner


def test_pipeline_runner_validation_failure_halts_early():
    runner = PipelineRunner()
    # Empty input fails Stage 1
    bundle = runner.run(user_input="   ")

    assert bundle.success is False
    assert "empty" in bundle.error_summary.lower()
    # Check that stage 1 failed and stage 2-5 were not completed
    assert bundle.stage_records["Stage 1: Input Processing"].status == StageStatus.FAILED
    assert bundle.stage_records["Stage 2: Prompt Engineering"].status == StageStatus.FAILED


@patch("pipeline.llm_client.OpenAI")
def test_pipeline_runner_full_successful_flow(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_choice = MagicMock()
    mock_choice.message.content = "Quantum mechanics allows qubits to superposition."
    mock_choice.finish_reason = "stop"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 30
    mock_usage.completion_tokens = 10
    mock_usage.total_tokens = 40

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model = "gpt-4o-mini"

    mock_client.chat.completions.create.return_value = mock_response

    runner = PipelineRunner()
    bundle = runner.run(
        user_input="Explain quantum computing in one sentence.",
        runtime_api_key="sk-dummy-test-key",
        model="gpt-4o-mini",
        style="Concise & Direct",
    )

    assert bundle.success is True
    assert bundle.error_summary is None
    assert bundle.final_output == "Quantum mechanics allows qubits to superposition."
    assert bundle.total_execution_time_seconds > 0

    # Ensure all 5 stages completed
    for stage_name, record in bundle.stage_records.items():
        assert record.status == StageStatus.COMPLETED, f"Stage {stage_name} did not complete"
        assert record.latency_seconds >= 0


@patch("pipeline.llm_client.OpenAI")
def test_pipeline_step_by_step_generator(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_choice = MagicMock()
    mock_choice.message.content = "Detailed explanation of recursion."
    mock_choice.finish_reason = "stop"
    mock_usage = MagicMock(prompt_tokens=20, completion_tokens=15, total_tokens=35)
    mock_response = MagicMock(choices=[mock_choice], usage=mock_usage, model="gpt-4o-mini")
    mock_client.chat.completions.create.return_value = mock_response

    runner = PipelineRunner()
    snapshots = list(
        runner.run_step_by_step(
            user_input="How does recursion work?",
            runtime_api_key="sk-test-key",
        )
    )

    # Should have multiple snapshots showing stage progression
    assert len(snapshots) >= 5
    last_snap = snapshots[-1]
    assert last_snap["bundle"] is not None
    assert last_snap["bundle"].success is True
