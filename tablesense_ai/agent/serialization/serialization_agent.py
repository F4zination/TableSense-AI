import pathlib
from typing import Union
import tiktoken

from tablesense_ai.agent.base import BaseAgent
from tablesense_ai.agent.serialization.converter import Converter, TableFormat
from tablesense_ai.utils.performance import measure_performance


class SerializationAgent(BaseAgent):
    """
    SerializationAgent is a specialized agent that handles the serialization of data.
    It extends the base Agent class and implements the eval method to perform serialization tasks.
    Because of API-Limitations, a token-limit was implemented
    """

    TOKEN_LIMIT = 5000

    def __init__(self, llm_model: str, temperature: float, max_retries: int,
                 max_tokens: int, base_url: str, api_key: str,
                 system_prompt: str = None,
                 format_to_convert_to: TableFormat = TableFormat.NATURAL, verbose: bool = False):
        super().__init__(llm_model, temperature, max_retries, max_tokens, base_url, api_key, system_prompt, verbose)
        self.converter = Converter()
        self.format_to_convert_to = format_to_convert_to
        self.encoding = tiktoken.get_encoding("o200k_base")

    @measure_performance
    def invoke(self, input_prompt: str) -> str:
        """
        Invoke the LLM with the given input.
        :param input_prompt:
        :return:
        """
        # Here you would implement the logic to invoke the LLM with the input
        # For example, using OpenAI's API or any other LLM service
        response = self.llm_model.invoke(input_prompt)
        return response

    def eval(self, question: str, dataset: pathlib.Path, additional_info: Union[dict, None]) -> str:
        """
        Evaluate the given data using the LLM.
        :param question:
        :param dataset:
        :param additional_info:
        :return:
        """
        # Perform the serialization task using the Converter class
        converted_content = self.converter.convert(dataset, self.format_to_convert_to, False)

        prompt = self.system_prompt.format(data=converted_content) + "\n" + question

        # Token-Limit
        token_count = len(self.encoding.encode(prompt))

        if token_count > self.TOKEN_LIMIT:
            if self.verbose:
                print(f"Prompt too long ({token_count} > {self.TOKEN_LIMIT} tokens).")
            return "skipped-too-long"

        # Use the LLM to generate a response based on the question and converted content
        response = self.invoke(prompt)

        return response.content
