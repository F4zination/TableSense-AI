import pathlib
from abc import ABC, abstractmethod
from langchain_openai import OpenAI


class Agent(ABC):

    def __init__(self, llm_model: str, temperature: float, max_retries: int, max_tokens: int, base_url: str,
                 api_key: str):
        self.llm_model = OpenAI(
            model=llm_model,
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
        )

    @abstractmethod
    def eval(self, question: str, dataset: pathlib.Path, additional_info: list[dict]) -> str:
        """
        Evaluate the given data using the LLM.

            :param question: The question to ask the LLM
            :param dataset: Path to the dataset for evaluation
            :param additional_info: Additional information to provide to the LLM (e.g., context, metadata)
            :return: The response from the LLM
        """
        pass
