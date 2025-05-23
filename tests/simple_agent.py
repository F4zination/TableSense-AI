# File: simple_agent.py
#from agent import Agent
from tablesense_ai.agent.base import BaseAgent

class SimpleAgent(BaseAgent):
    def __init__(self, llm_model: str, temperature: float, max_retries: int, max_tokens: int,
                 base_url: str, api_key: str, system_prompt: str = None):
        super().__init__(llm_model, temperature, max_retries, max_tokens, base_url, api_key, system_prompt)

    def eval(self, question: str, dataset=None, additional_info=None) -> str:
        messages = [
            {"role": "system", "content": f"{self.system_prompt}"},
            {"role": "user", "content": f"{question}"},
        ]
        response = self.llm_model.invoke(input=messages)
        return response

