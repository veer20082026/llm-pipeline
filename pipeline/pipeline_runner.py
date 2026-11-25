"""
Pipeline Runner: Orchestrates all 5 stages of the LLM Pipeline.

Provides stage-by-stage generator iteration so frontends (Streamlit / CLI)
can display real-time status transitions (Pending -> Running -> Completed/Failed),
intermediate artifacts, and logs.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generator, List, Optional

from config.settings import AppConfig, PipelineStage, StageStatus, get_config
from pipeline.input_processor import InputProcessingResult, InputProcessor
from pipeline.llm_client import LLMClient, LLMIntegrationResult
from pipeline.output_handler import OutputHandler, PipelineExecutionBundle, StageRecord
from pipeline.prompt_builder import PromptBuilder, PromptEngineeringResult
from pipeline.response_processor import ResponseProcessingResult, ResponseProcessor


class PipelineRunner:
    """
    Coordinates execution of all five pipeline stages with comprehensive logging
    and status reporting.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or get_config()
        self.input_processor = InputProcessor(
            min_length=self.config.min_input_length,
            max_length=self.config.max_input_length,
        )
        self.prompt_builder = PromptBuilder()
        self.llm_client = LLMClient(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.request_timeout,
        )
        self.response_processor = ResponseProcessor()
        self.output_handler = OutputHandler()
        self.logs: List[str] = []

    def log(self, message: str) -> None:
        """Record a timestamped pipeline event."""
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{now_str}] {message}"
        self.logs.append(entry)

    def run_step_by_step(
        self,
        user_input: str,
        model: str = "gpt-4o-mini",
        style: str = "Concise & Direct",
        persona: str = "General Assistant",
        custom_system_instruction: Optional[str] = None,
        task_context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        runtime_api_key: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, PipelineExecutionBundle]:
        """
        Generator executing all five pipeline stages sequentially.
        Yields progress snapshots after each event:
        {
            "current_stage": PipelineStage,
            "stage_status": StageStatus,
            "stage_records": Dict[str, StageRecord],
            "logs": List[str],
            "bundle": Optional[PipelineExecutionBundle]
        }
        """
        self.logs.clear()
        total_start = time.perf_counter()

        # Initialize all 5 stage records as PENDING
        stages = [
            PipelineStage.INPUT_PROCESSING,
            PipelineStage.PROMPT_ENGINEERING,
            PipelineStage.LLM_INTEGRATION,
            PipelineStage.RESPONSE_PROCESSING,
            PipelineStage.OUTPUT_GENERATION,
        ]
        stage_records: Dict[str, StageRecord] = {
            s.value: StageRecord(stage=s, status=StageStatus.PENDING)
            for s in stages
        }

        input_res: Optional[InputProcessingResult] = None
        prompt_res: Optional[PromptEngineeringResult] = None
        llm_res: Optional[LLMIntegrationResult] = None
        response_res: Optional[ResponseProcessingResult] = None
        error_summary: Optional[str] = None

        def create_snapshot(cur_stage: PipelineStage, cur_status: StageStatus) -> Dict[str, Any]:
            return {
                "current_stage": cur_stage,
                "stage_status": cur_status,
                "stage_records": stage_records,
                "logs": list(self.logs),
                "input_result": input_res,
                "prompt_result": prompt_res,
                "llm_result": llm_res,
                "response_result": response_res,
                "bundle": None,
            }

        # ---------------------------------------------------------------------
        # STAGE 1: INPUT PROCESSING
        # ---------------------------------------------------------------------
        s1 = PipelineStage.INPUT_PROCESSING.value
        stage_records[s1].status = StageStatus.RUNNING
        self.log(f"Starting {s1}...")
        yield create_snapshot(PipelineStage.INPUT_PROCESSING, StageStatus.RUNNING)

        s1_start = time.perf_counter()
        input_res = self.input_processor.process(user_input)
        stage_records[s1].latency_seconds = time.perf_counter() - s1_start
        stage_records[s1].details = input_res.to_dict()

        if not input_res.is_valid:
            error_summary = input_res.validation_error or "Input validation failed."
            stage_records[s1].status = StageStatus.FAILED
            stage_records[s1].error_message = error_summary
            self.log(f"[ERROR] {s1} failed: {error_summary}")
            yield create_snapshot(PipelineStage.INPUT_PROCESSING, StageStatus.FAILED)

            # Mark subsequent stages as failed/cancelled
            for remaining in stages[1:]:
                stage_records[remaining.value].status = StageStatus.FAILED
                stage_records[remaining.value].error_message = "Skipped due to earlier stage failure."

            total_elapsed = time.perf_counter() - total_start
            bundle = self.output_handler.generate_bundle(
                original_input=user_input,
                stage_records=stage_records,
                input_result=input_res,
                model_used=model,
                style_applied=style,
                total_time=total_elapsed,
                error_summary=error_summary,
            )
            final_snap = create_snapshot(PipelineStage.INPUT_PROCESSING, StageStatus.FAILED)
            final_snap["bundle"] = bundle
            yield final_snap
            return bundle

        stage_records[s1].status = StageStatus.COMPLETED
        self.log(
            f"{s1} completed successfully in {round(stage_records[s1].latency_seconds, 3)}s. "
            f"Words: {input_res.word_count}, Chars: {input_res.character_count}, Est. tokens: {input_res.estimated_tokens}"
        )
        yield create_snapshot(PipelineStage.INPUT_PROCESSING, StageStatus.COMPLETED)

        # ---------------------------------------------------------------------
        # STAGE 2: PROMPT ENGINEERING
        # ---------------------------------------------------------------------
        s2 = PipelineStage.PROMPT_ENGINEERING.value
        stage_records[s2].status = StageStatus.RUNNING
        self.log(f"Starting {s2} with style='{style}' and persona='{persona}'...")
        yield create_snapshot(PipelineStage.PROMPT_ENGINEERING, StageStatus.RUNNING)

        s2_start = time.perf_counter()
        try:
            prompt_res = self.prompt_builder.build(
                cleaned_input=input_res.cleaned_text,
                style_name=style,
                persona_name=persona,
                custom_system_instruction=custom_system_instruction,
                task_context=task_context,
            )
            stage_records[s2].latency_seconds = time.perf_counter() - s2_start
            stage_records[s2].details = prompt_res.to_dict()
            stage_records[s2].status = StageStatus.COMPLETED
            self.log(
                f"{s2} completed in {round(stage_records[s2].latency_seconds, 3)}s. "
                f"Total prompt chars: {prompt_res.total_prompt_length}, Est. prompt tokens: {prompt_res.estimated_prompt_tokens}"
            )
            yield create_snapshot(PipelineStage.PROMPT_ENGINEERING, StageStatus.COMPLETED)
        except Exception as exc:
            error_summary = f"Prompt building failed: {exc}"
            stage_records[s2].latency_seconds = time.perf_counter() - s2_start
            stage_records[s2].status = StageStatus.FAILED
            stage_records[s2].error_message = error_summary
            self.log(f"[ERROR] {s2} failed: {exc}")
            for remaining in stages[2:]:
                stage_records[remaining.value].status = StageStatus.FAILED
                stage_records[remaining.value].error_message = "Skipped due to earlier stage failure."

            total_elapsed = time.perf_counter() - total_start
            bundle = self.output_handler.generate_bundle(
                original_input=user_input,
                stage_records=stage_records,
                input_result=input_res,
                prompt_result=prompt_res,
                model_used=model,
                style_applied=style,
                total_time=total_elapsed,
                error_summary=error_summary,
            )
            final_snap = create_snapshot(PipelineStage.PROMPT_ENGINEERING, StageStatus.FAILED)
            final_snap["bundle"] = bundle
            yield final_snap
            return bundle

        # ---------------------------------------------------------------------
        # STAGE 3: LLM INTEGRATION
        # ---------------------------------------------------------------------
        s3 = PipelineStage.LLM_INTEGRATION.value
        stage_records[s3].status = StageStatus.RUNNING
        self.log(f"Starting {s3}: Invoking model='{model}' (temp={temperature}, max_tokens={max_tokens})...")
        yield create_snapshot(PipelineStage.LLM_INTEGRATION, StageStatus.RUNNING)

        llm_res = self.llm_client.generate(
            messages=prompt_res.messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            runtime_api_key=runtime_api_key,
        )

        stage_records[s3].latency_seconds = llm_res.latency_seconds
        stage_records[s3].details = llm_res.to_dict()

        if not llm_res.success:
            error_summary = llm_res.error_message or "LLM generation failed."
            stage_records[s3].status = StageStatus.FAILED
            stage_records[s3].error_message = error_summary
            self.log(f"[ERROR] {s3} failed: {error_summary}")
            for remaining in stages[3:]:
                stage_records[remaining.value].status = StageStatus.FAILED
                stage_records[remaining.value].error_message = "Skipped due to earlier stage failure."

            total_elapsed = time.perf_counter() - total_start
            bundle = self.output_handler.generate_bundle(
                original_input=user_input,
                stage_records=stage_records,
                input_result=input_res,
                prompt_result=prompt_res,
                llm_result=llm_res,
                model_used=model,
                style_applied=style,
                total_time=total_elapsed,
                error_summary=error_summary,
            )
            final_snap = create_snapshot(PipelineStage.LLM_INTEGRATION, StageStatus.FAILED)
            final_snap["bundle"] = bundle
            yield final_snap
            return bundle

        stage_records[s3].status = StageStatus.COMPLETED
        self.log(
            f"{s3} finished in {round(llm_res.latency_seconds, 3)}s. "
            f"Tokens: prompt={llm_res.prompt_tokens}, completion={llm_res.completion_tokens}, total={llm_res.total_tokens}"
        )
        yield create_snapshot(PipelineStage.LLM_INTEGRATION, StageStatus.COMPLETED)

        # ---------------------------------------------------------------------
        # STAGE 4: RESPONSE PROCESSING
        # ---------------------------------------------------------------------
        s4 = PipelineStage.RESPONSE_PROCESSING.value
        stage_records[s4].status = StageStatus.RUNNING
        self.log(f"Starting {s4}: Cleaning response, analyzing structure & metrics...")
        yield create_snapshot(PipelineStage.RESPONSE_PROCESSING, StageStatus.RUNNING)

        s4_start = time.perf_counter()
        response_res = self.response_processor.process(
            raw_response=llm_res.raw_response,
            completion_tokens=llm_res.completion_tokens,
            latency_seconds=llm_res.latency_seconds,
        )
        stage_records[s4].latency_seconds = time.perf_counter() - s4_start
        stage_records[s4].details = response_res.to_dict()

        if not response_res.is_valid:
            error_summary = response_res.error_message or "Response processing failed."
            stage_records[s4].status = StageStatus.FAILED
            stage_records[s4].error_message = error_summary
            self.log(f"[ERROR] {s4} failed: {error_summary}")
            stage_records[PipelineStage.OUTPUT_GENERATION.value].status = StageStatus.FAILED
            stage_records[PipelineStage.OUTPUT_GENERATION.value].error_message = "Skipped."

            total_elapsed = time.perf_counter() - total_start
            bundle = self.output_handler.generate_bundle(
                original_input=user_input,
                stage_records=stage_records,
                input_result=input_res,
                prompt_result=prompt_res,
                llm_result=llm_res,
                response_result=response_res,
                model_used=model,
                style_applied=style,
                total_time=total_elapsed,
                error_summary=error_summary,
            )
            final_snap = create_snapshot(PipelineStage.RESPONSE_PROCESSING, StageStatus.FAILED)
            final_snap["bundle"] = bundle
            yield final_snap
            return bundle

        stage_records[s4].status = StageStatus.COMPLETED
        self.log(
            f"{s4} completed in {round(stage_records[s4].latency_seconds, 3)}s. "
            f"Words: {response_res.word_count}, Reading time: ~{response_res.estimated_reading_time_minutes}m, "
            f"Speed: {response_res.tokens_per_second} tokens/sec"
        )
        yield create_snapshot(PipelineStage.RESPONSE_PROCESSING, StageStatus.COMPLETED)

        # ---------------------------------------------------------------------
        # STAGE 5: OUTPUT GENERATION
        # ---------------------------------------------------------------------
        s5 = PipelineStage.OUTPUT_GENERATION.value
        stage_records[s5].status = StageStatus.RUNNING
        self.log(f"Starting {s5}: Packaging artifacts and preparing output...")
        yield create_snapshot(PipelineStage.OUTPUT_GENERATION, StageStatus.RUNNING)

        s5_start = time.perf_counter()
        total_elapsed = time.perf_counter() - total_start
        stage_records[s5].latency_seconds = time.perf_counter() - s5_start
        stage_records[s5].status = StageStatus.COMPLETED

        bundle = self.output_handler.generate_bundle(
            original_input=user_input,
            stage_records=stage_records,
            input_result=input_res,
            prompt_result=prompt_res,
            llm_result=llm_res,
            response_result=response_res,
            model_used=model,
            style_applied=style,
            total_time=total_elapsed,
            error_summary=None,
        )
        stage_records[s5].details = {
            "total_latency_seconds": round(total_elapsed, 3),
            "output_length": len(bundle.final_output),
        }

        self.log(f"{s5} completed in {round(stage_records[s5].latency_seconds, 3)}s.")
        self.log(f"All 5 stages completed successfully in {round(total_elapsed, 3)}s total.")

        final_snap = create_snapshot(PipelineStage.OUTPUT_GENERATION, StageStatus.COMPLETED)
        final_snap["bundle"] = bundle
        yield final_snap

        return bundle

    def run(
        self,
        user_input: str,
        model: str = "gpt-4o-mini",
        style: str = "Concise & Direct",
        persona: str = "General Assistant",
        custom_system_instruction: Optional[str] = None,
        task_context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        runtime_api_key: Optional[str] = None,
    ) -> PipelineExecutionBundle:
        """Synchronously execute the full pipeline to completion."""
        generator = self.run_step_by_step(
            user_input=user_input,
            model=model,
            style=style,
            persona=persona,
            custom_system_instruction=custom_system_instruction,
            task_context=task_context,
            temperature=temperature,
            max_tokens=max_tokens,
            runtime_api_key=runtime_api_key,
        )
        last_snap = None
        for snapshot in generator:
            last_snap = snapshot

        if last_snap and last_snap.get("bundle"):
            return last_snap["bundle"]

        total_elapsed = 0.0
        return self.output_handler.generate_bundle(
            original_input=user_input,
            stage_records={},
            error_summary="Pipeline terminated unexpectedly.",
        )
