"""
Token Budgeting, Calculation & Cost Estimation (Topic 3.14, 3.15)
"""
import tiktoken
from typing import Dict


class TokenManager:
    # Approximate pricing per 1M tokens for gpt-4o-mini
    INPUT_COST_PER_M = 0.150  # USD
    OUTPUT_COST_PER_M = 0.600  # USD

    def __init__(self, model_name: str = "gpt-4o-mini"):
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> Dict[str, float]:
        input_cost = (prompt_tokens / 1_000_000) * self.INPUT_COST_PER_M
        output_cost = (completion_tokens / 1_000_000) * self.OUTPUT_COST_PER_M
        return {
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(input_cost + output_cost, 6),
        }