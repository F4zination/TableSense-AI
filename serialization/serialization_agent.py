import pathlib
from agent import Agent
from serialization.converter import Converter, FileFormat


class SerializationAgent(Agent):
    """
    SerializationAgent is a specialized agent that handles the serialization of data.
    It extends the base Agent class and implements the eval method to perform serialization tasks.
    """

    def __init__(self, llm_model: str, temperature: float, max_retries: int, max_tokens: int, base_url: str,
                 api_key: str):
        super().__init__(llm_model, temperature, max_retries, max_tokens, base_url, api_key)
        self.converter = Converter()


    def eval(self, question: str, dataset: pathlib.Path, additional_info: list[dict]) -> str:
        """
        Evaluate the given data using the LLM.
        :param question:
        :param dataset:
        :param additional_info:
        :return:
        """
        # Perform the serialization task using the Converter class
        converted_content = self.converter.convert(dataset, FileFormat.HTML, False)

        # Use the LLM to generate a response based on the question and converted content
        response = self.llm_model.invoke(question + converted_content)

        return response

