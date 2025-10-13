import pathlib
from typing import Union, List, Dict, Any
import tiktoken
from openai import OpenAI

from tablesense_ai.agent.base import BaseAgent
from tablesense_ai.agent.serialization.converter import Converter, TableFormat
from tablesense_ai.utils.performance import measure_performance


# Two curated, reusable system prompts
SYSTEM_PROMPT = (
    "You are an expert data analyst.\n"
    "Use only the table(s) provided in the user message.\n"
    "Answer by calculating or copying values from the provided table\n"
    "Keep the keep the format provided in the table (case, accents, punctuation, separators, date/time formats)."
    "If the answer is missing or cannot be inferred, output exactly 'N/A'.\n"
    "Return only the final answer string — no labels, no quotes, no code, no explanations.\n"
)


class ChatResult:
    """Minimal wrapper to normalize content and token usage for performance logging."""

    def __init__(self, content: str, prompt_tokens: int | None, completion_tokens: int | None):
        self.content = content
        self.response_metadata = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        }


class SerializationAgent(BaseAgent):
    """
    SerializationAgent is a specialized agent that handles the serialization of data.
    It extends the base Agent class and implements the eval method to perform serialization tasks.
    Because of API-Limitations, a token-limit was implemented
    """

    TOKEN_LIMIT = 32700

    def __init__(self, llm_model: str, temperature: float, max_retries: int,
                 max_tokens: int, base_url: str, api_key: str,
                 system_prompt: str = None,
                 format_to_convert_to: TableFormat = TableFormat.NATURAL, verbose: bool = False):
        super().__init__(llm_model, temperature, max_retries, max_tokens, base_url, api_key, system_prompt, verbose)
        self.converter = Converter()
        self.format_to_convert_to = format_to_convert_to
        self.encoding = tiktoken.get_encoding("o200k_base")
        # Prepare OpenAI client for Chat Completions
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        # Use provided system_prompt if any; otherwise default to the curated profile
        self.system_prompt = system_prompt if system_prompt else SYSTEM_PROMPT

    @measure_performance
    def invoke(self, messages: List[Dict[str, str]]) -> ChatResult:
        """
        Call OpenAI-compatible Chat Completions API with structured messages.
        Returns a ChatResult with content and usage for perf logging.
        """
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # [{"role": "system"|"user", "content": str}, ...]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        print(messages)
        content = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        return ChatResult(content, prompt_tokens, completion_tokens)

    def eval(self, question: str, dataset: pathlib.Path, dataset_prompt: Union[str, dict, None]) -> str:
        """
        Evaluate the given data using the LLM.
        :param question:
        :param dataset:
        :param dataset_prompt:
        :return:
        """
        # 1) Convert the table to the requested format
        converted_content = self.converter.convert(dataset, self.format_to_convert_to, False)

        # 2) Build user message with clear table markers
        user_message = self._build_user_message(
            question=question,
            table_content=converted_content,
            table_format=self.format_to_convert_to,
            dataset_prompt=dataset_prompt,
        )

        # 3) Assemble messages with a stable system prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        # 4) Token-limit check (approximate: count system+user concatenated)
        joined = self.system_prompt + "\n\n" + user_message
        token_count = len(self.encoding.encode(joined))
        if token_count > self.TOKEN_LIMIT:
            if self.verbose:
                print(f"Prompt too long ({token_count} > {self.TOKEN_LIMIT} tokens).")
            return "skipped-too-long"

        # 5) Call chat completions and return the answer string
        response = self.invoke(messages)
        return response.content

    def _build_user_message(
        self,
        question: str,
        table_content: str,
        table_format: TableFormat,
        dataset_prompt: Union[str, dict, None] = None,
    ) -> str:
        """Create a clean user message: question + fenced table + optional notes."""
        parts: list[str] = []
        parts.append("You are given one or more tables. Answer using only these tables.")
        if dataset_prompt:
            # Accept dict or str and normalize to one short notes block
            if isinstance(dataset_prompt, dict):
                try:
                    info = "; ".join(f"{k}: {v}" for k, v in dataset_prompt.items())
                except Exception:
                    info = str(dataset_prompt)
            else:
                info = str(dataset_prompt)
            parts.append(f"Notes: {info}")

        parts.append(f"Question: {question}")
        parts.append(f"[BEGIN TABLE #1 — format: {table_format.name}]")
        parts.append(table_content)
        parts.append("[END TABLE #1]")

        return "\n".join(parts)
