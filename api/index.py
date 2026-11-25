"""
Vercel Serverless Function entrypoint for LLM Pipeline.

Exports `app` and `handler` for Vercel's Python runtime.
Handles /api/run requests and executes the 5-stage pipeline runner.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# Ensure project root is on sys.path so pipeline and config modules resolve
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import AppConfig
from pipeline.pipeline_runner import PipelineRunner


class handler(BaseHTTPRequestHandler):
    """Vercel HTTP serverless handler for LLM pipeline executions."""

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight requests."""
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        """Health check endpoint."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        response_data = {
            "status": "healthy",
            "service": "LLM Pipeline Serverless API",
            "stages": [
                "1. Input Processing",
                "2. Prompt Engineering",
                "3. LLM Integration",
                "4. Response Processing",
                "5. Output Generation",
            ],
        }
        self.wfile.write(json.dumps(response_data).encode("utf-8"))

    def do_POST(self) -> None:
        """Execute the complete multi-stage LLM pipeline."""
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(
                json.dumps({"success": False, "error": f"Invalid JSON payload: {e}"}).encode("utf-8")
            )
            return

        user_input = payload.get("user_input", "")
        model = payload.get("model", "gpt-4o-mini")
        style = payload.get("style", "Concise & Direct")
        persona = payload.get("persona", "General Assistant")
        custom_system = payload.get("custom_system_instruction")
        task_context = payload.get("task_context")
        temperature = float(payload.get("temperature", 0.7))
        max_tokens = int(payload.get("max_tokens", 1500))
        runtime_api_key = payload.get("api_key") or os.getenv("OPENAI_API_KEY", "")
        base_url = payload.get("base_url") or os.getenv("OPENAI_BASE_URL", "")

        # Initialize config & runner
        config = AppConfig(
            api_key=runtime_api_key,
            base_url=base_url if base_url.strip() else None,
        )
        runner = PipelineRunner(config=config)

        # Execute pipeline
        bundle = runner.run(
            user_input=user_input,
            model=model,
            style=style,
            persona=persona,
            custom_system_instruction=custom_system,
            task_context=task_context,
            temperature=temperature,
            max_tokens=max_tokens,
            runtime_api_key=runtime_api_key,
        )

        response_payload = bundle.to_dict()
        response_payload["logs"] = runner.logs

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(response_payload).encode("utf-8"))


# Export `app` alias so Vercel can find either `app` or `handler`
app = handler
