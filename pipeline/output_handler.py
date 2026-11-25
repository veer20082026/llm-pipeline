"""
Stage 5: Output Generation Module.

Assembles the complete end-to-end pipeline execution package, aggregates
metrics, formats outputs for visual presentation, clipboard copy, and file exports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config.settings import PipelineStage, StageStatus
from pipeline.input_processor import InputProcessingResult
from pipeline.prompt_builder import PromptEngineeringResult
from pipeline.llm_client import LLMIntegrationResult
from pipeline.response_processor import ResponseProcessingResult


@dataclass
class StageRecord:
    """Record of a single stage's execution within the pipeline."""
    stage: PipelineStage
    status: StageStatus = StageStatus.PENDING
    latency_seconds: float = 0.0
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "latency_seconds": round(self.latency_seconds, 3),
            "error_message": self.error_message,
            "details": self.details,
        }


@dataclass
class PipelineExecutionBundle:
    """Comprehensive artifact produced upon pipeline completion."""
    success: bool
    timestamp: str
    original_input: str
    final_output: str
    model_used: str
    style_applied: str
    total_execution_time_seconds: float
    stage_records: Dict[str, StageRecord]
    input_result: Optional[InputProcessingResult] = None
    prompt_result: Optional[PromptEngineeringResult] = None
    llm_result: Optional[LLMIntegrationResult] = None
    response_result: Optional[ResponseProcessingResult] = None
    error_summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert full pipeline execution to dictionary for export."""
        return {
            "success": self.success,
            "timestamp": self.timestamp,
            "model_used": self.model_used,
            "style_applied": self.style_applied,
            "total_execution_time_seconds": round(self.total_execution_time_seconds, 3),
            "error_summary": self.error_summary,
            "stages": {k: v.to_dict() for k, v in self.stage_records.items()},
            "input_metrics": self.input_result.to_dict() if self.input_result else None,
            "prompt_metrics": self.prompt_result.to_dict() if self.prompt_result else None,
            "llm_metrics": self.llm_result.to_dict() if self.llm_result else None,
            "response_metrics": self.response_result.to_dict() if self.response_result else None,
        }


class OutputHandler:
    """
    Stage 5 Handler: Formats final outputs, comparison displays, and export bundles.
    """

    def generate_bundle(
        self,
        original_input: str,
        stage_records: Dict[str, StageRecord],
        input_result: Optional[InputProcessingResult] = None,
        prompt_result: Optional[PromptEngineeringResult] = None,
        llm_result: Optional[LLMIntegrationResult] = None,
        response_result: Optional[ResponseProcessingResult] = None,
        model_used: str = "gpt-4o-mini",
        style_applied: str = "Concise & Direct",
        total_time: float = 0.0,
        error_summary: Optional[str] = None,
    ) -> PipelineExecutionBundle:
        """Compile all stage artifacts into the definitive execution bundle."""
        final_text = response_result.processed_response if response_result else ""
        is_success = bool(
            input_result and input_result.is_valid
            and prompt_result
            and llm_result and llm_result.success
            and response_result and response_result.is_valid
            and not error_summary
        )

        return PipelineExecutionBundle(
            success=is_success,
            timestamp=datetime.now(timezone.utc).isoformat(),
            original_input=original_input,
            final_output=final_text,
            model_used=model_used,
            style_applied=style_applied,
            total_execution_time_seconds=total_time,
            stage_records=stage_records,
            input_result=input_result,
            prompt_result=prompt_result,
            llm_result=llm_result,
            response_result=response_result,
            error_summary=error_summary,
        )

    def format_as_text(self, bundle: PipelineExecutionBundle) -> str:
        """Format the execution bundle as a clean, human-readable plain text file."""
        divider = "=" * 60
        subdivider = "-" * 60

        lines = [
            divider,
            "             LLM PIPELINE EXECUTION REPORT",
            divider,
            f"Timestamp:          {bundle.timestamp}",
            f"Overall Status:     {'SUCCESS' if bundle.success else 'FAILED'}",
            f"Model Used:         {bundle.model_used}",
            f"Output Style:       {bundle.style_applied}",
            f"Total Latency:      {round(bundle.total_execution_time_seconds, 3)} seconds",
        ]

        if bundle.llm_result:
            lines.extend([
                f"Prompt Tokens:      {bundle.llm_result.prompt_tokens}",
                f"Completion Tokens:  {bundle.llm_result.completion_tokens}",
                f"Total Tokens:       {bundle.llm_result.total_tokens}",
            ])

        if bundle.response_result and bundle.response_result.tokens_per_second > 0:
            lines.append(f"Generation Speed:   {bundle.response_result.tokens_per_second} tokens/sec")

        if bundle.error_summary:
            lines.extend([
                subdivider,
                f"ERROR: {bundle.error_summary}",
            ])

        lines.extend([
            subdivider,
            "STAGE BREAKDOWN:",
            subdivider,
        ])

        for stage_name, record in bundle.stage_records.items():
            status_icon = (
                "[PASS]" if record.status == StageStatus.COMPLETED
                else "[FAIL]" if record.status == StageStatus.FAILED
                else f"[{record.status.value.upper()}]"
            )
            lines.append(
                f"{status_icon} {stage_name} ({round(record.latency_seconds, 3)}s)"
            )
            if record.error_message:
                lines.append(f"       Error: {record.error_message}")

        lines.extend([
            divider,
            "ORIGINAL USER INPUT:",
            divider,
            bundle.original_input,
            "",
            divider,
            "FINAL GENERATED OUTPUT:",
            divider,
            bundle.final_output if bundle.final_output else "(No output generated)",
            divider,
        ])

        return "\n".join(lines)

    def format_as_json(self, bundle: PipelineExecutionBundle) -> str:
        """Format the execution bundle as indented JSON."""
        return json.dumps(bundle.to_dict(), indent=2)
