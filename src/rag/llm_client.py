"""
LLM Client Wrapper with Retry Strategies and Schema Enforcement (Topic 3.12, 3.16, 3.17)
"""
import os
import json
import time
from typing import Dict, Any, Tuple
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from dotenv import load_dotenv

from src.rag.schemas import DiagnosticResponse
from src.rag.prompts import load_system_prompt, build_technician_prompt
from src.rag.token_budget import TokenManager

load_dotenv()


class DiagnosticLLMClient:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("CHAT_MODEL", "gpt-4o-mini")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured in .env")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.token_manager = TokenManager(model_name=self.model)

    def run_completion(
        self,
        vehicle_model: str,
        model_year: int,
        variant: str,
        region: str,
        query: str,
        retrieved_context: str = "",
        max_retries: int = 3
    ) -> Tuple[DiagnosticResponse, Dict[str, Any]]:
        system_prompt = load_system_prompt()
        user_prompt = build_technician_prompt(
            vehicle_model=vehicle_model,
            model_year=model_year,
            variant=variant,
            region=region,
            query=query,
            retrieved_context=retrieved_context
        )

        prompt_tokens = (
            self.token_manager.count_tokens(system_prompt) +
            self.token_manager.count_tokens(user_prompt)
        )

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format=DiagnosticResponse,
                    temperature=0.0,
                    timeout=30.0
                )

                parsed_output: DiagnosticResponse = response.choices[0].message.parsed
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                cost_metrics = self.token_manager.estimate_cost(prompt_tokens, completion_tokens)

                telemetry = {
                    "model": self.model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost_metrics": cost_metrics,
                    "status": "SUCCESS"
                }

                return parsed_output, telemetry

            except (RateLimitError, APITimeoutError) as transient_err:
                if attempt == max_retries:
                    raise RuntimeError(f"LLM API failed after {max_retries} attempts: {str(transient_err)}")
                time.sleep(2 ** attempt)  # Exponential backoff

            except APIError as api_err:
                raise RuntimeError(f"OpenAI API error encountered: {str(api_err)}")