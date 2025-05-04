# File: simple_agent.py
from agent import Agent


class SimpleAgent(Agent):
    def __init__(self, llm_model: str, temperature: float, max_retries: int, max_tokens: int,
                 base_url: str, api_key: str, system_prompt: str = None):
        super().__init__(llm_model, temperature, max_retries, max_tokens, base_url, api_key, system_prompt)

    def eval(self, question: str, dataset=None, additional_info=None) -> str:
        prompt = f"{self.system_prompt}\n{question}"
        print(f"Generated Prompt: {prompt}")  # Debugging: Prompt anzeigen
        response = self.llm_model.invoke(prompt)
        if not response:  # Prüfen, ob die Antwort leer ist
            print("Error: LLM response is empty.")
        return response

