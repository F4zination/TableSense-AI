import pathlib
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
import time
import tracemalloc
from functools import wraps

default_prompt = (
    "You are acting as an expert data analyst.\n\n"
    "You are provided with a table as a pandas DataFrame, shown as {data}.\n\n"
    "Your role is to respond to user questions about this dataset.\n\n"
    "For each user question, analyze and interpret the data in the DataFrame as needed.\n\n"
    "Do not provide code, demonstrations, or step-by-step explanations; instead, directly answer the user's question.\n\n"
    "Only return the requested answer to the question, nothing more!\n\n"
    "Only refer to the data available in the DataFrame ({data}) when constructing your answer.\n\n"
    "If a question requires numerical results (e.g., averages, sums), provide the computed figure.\n\n"
    "Assume each question stands on its own; do not reference previous questions or context beyond the current input.\n\n"
    "Your output should always be a single string with no code, comments, or formatting syntax.\n\n"
    "Focus on precision and informativeness in your response. Always communicate clearly and avoid unnecessary detail.\n\n"
    "Keep your answer AS SHORT AS POSSIBLE."
)

class BaseAgent(ABC):

    def __init__(self, llm_model: str, temperature: float, max_retries: int, max_tokens: int, base_url: str,
                 api_key: str, system_prompt: str = None, verbose: bool = False):
        self.llm_model = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
        )
        self.system_prompt = system_prompt if system_prompt else default_prompt
        self.verbose = verbose




    def set_system_prompt(self, system_prompt: str):
        """
        Set the system prompt for the LLM.

            :param system_prompt: The system prompt to set
        """
        self.system_prompt = system_prompt

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
