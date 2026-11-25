"""
Stage 3: LLM Integration Module.

Connects to actual OpenAI or OpenAI-compatible model endpoints (Groq, Together,
DeepSeek, Local Ollama, OpenAI) via the standard OpenAI Python SDK.
Handles authentication, rate limits, latency timing, and API errors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import openai
from openai import OpenAI


@dataclass
class LLMIntegrationResult:
    """Represents the output of Stage 3: LLM Integration."""
    success: bool
    raw_response: str
    model: str
    latency_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization and logging."""
        return {
            "success": self.success,
            "raw_response": self.raw_response,
            "model": self.model,
            "latency_seconds": round(self.latency_seconds, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class LLMClient:
    """
    Stage 3 Client: Manages interactions with OpenAI-compatible LLM endpoints.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.strip() if (base_url and base_url.strip()) else None
        self.timeout = timeout
        self._client: Optional[OpenAI] = None

    def _get_client(self, runtime_api_key: Optional[str] = None) -> OpenAI:
        """Instantiate or retrieve the OpenAI client."""
        effective_key = runtime_api_key or self.api_key
        if not effective_key:
            raise ValueError(
                "API key is missing. Please set OPENAI_API_KEY in your .env file "
                "or enter it in the sidebar configuration."
            )

        kwargs: Dict[str, Any] = {
            "api_key": effective_key,
            "timeout": float(self.timeout),
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url

        return OpenAI(**kwargs)

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1500,
        runtime_api_key: Optional[str] = None,
    ) -> LLMIntegrationResult:
        """
        Execute Stage 3: Send chat completion request to the LLM API.
        Measures exact round-trip latency and captures usage tokens.
        """
        start_time = time.perf_counter()

        try:
            client = self._get_client(runtime_api_key=runtime_api_key)

            # Send request to the OpenAI-compatible endpoint
            response = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
            )

            latency = time.perf_counter() - start_time

            choice = response.choices[0]
            raw_text = choice.message.content or ""
            finish_reason = choice.finish_reason

            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0

            return LLMIntegrationResult(
                success=True,
                raw_response=raw_text,
                model=response.model or model,
                latency_seconds=latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason=finish_reason,
                metadata={
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                    "created": getattr(response, "created", None),
                },
            )

        except openai.AuthenticationError as err:
            latency = time.perf_counter() - start_time
            return LLMIntegrationResult(
                success=False,
                raw_response="",
                model=model,
                latency_seconds=latency,
                error_message=(
                    f"Authentication Error: Invalid API key or unauthorized access. ({err})"
                ),
            )

        except openai.RateLimitError as err:
            latency = time.perf_counter() - start_time
            return LLMIntegrationResult(
                success=False,
                raw_response="",
                model=model,
                latency_seconds=latency,
                error_message=(
                    f"Rate Limit Exceeded: The API rate limit or quota has been reached. ({err})"
                ),
            )

        except openai.APIConnectionError as err:
            latency = time.perf_counter() - start_time
            return LLMIntegrationResult(
                success=False,
                raw_response="",
                model=model,
                latency_seconds=latency,
                error_message=(
                    f"Connection Error: Failed to reach the LLM endpoint at '{self.base_url or 'api.openai.com'}'. "
                    f"Check your internet connection or base URL. ({err})"
                ),
            )

        except openai.BadRequestError as err:
            latency = time.perf_counter() - start_time
            return LLMIntegrationResult(
                success=False,
                raw_response="",
                model=model,
                latency_seconds=latency,
                error_message=(
                    f"Bad Request: The request was invalid or exceeded model token limits. ({err})"
                ),
            )

        except openai.APIError as err:
            latency = time.perf_counter() - start_time
            return LLMIntegrationResult(
                success=False,
                raw_response="",
                model=model,
                latency_seconds=latency,
                error_message=f"OpenAI API Error: {err}",
            )

        except ValueError as err:
            latency = time.perf_counter() - start_time
            return LLMIntegrationResult(
                success=False,
                raw_response="",
                model=model,
                latency_seconds=latency,
                error_message=str(err),
            )

        except Exception as err:
            latency = time.perf_counter() - start_time
            return LLMIntegrationResult(
                success=False,
                raw_response="",
                model=model,
                latency_seconds=latency,
                error_message=f"Unexpected error during LLM invocation: {err}",
            )
