# File: test_llm_exact_match.py
import unittest
from tests.simple_agent import SimpleAgent
from evaluate import load


class TestLLMExactMatch(unittest.TestCase):
    def setUp(self):
        """
        Set up the LLM agent and exact match metric for evaluation.
        """
        # Configure the agent with the real LLM
        self.agent = SimpleAgent(
            llm_model="/models/mistral-nemo-12b",
            temperature=0,
            max_retries=2,
            max_tokens=2048,
            base_url="http://80.151.131.52:9180/v1",
            api_key="THU-I17468S973-Student-24-25-94682Y1315",
            system_prompt="Answer the question as short as possible!!!"
        )

        # Load the exact match metric
        self.exact_match_metric = load("exact_match")

    def test_llm_with_exact_match(self):
        """
        Test LLM responses to 10 basic questions using exact match.
        """
        import time
        # Define the questions and ground-truth answers
        test_data = [
            {
                "id": "1",
                "utterance": "What is the capital of France?",
                "target_value": "Paris",
                "context": {"csv": "", "html": "", "tsv": ""}
            },
            {
                "id": "2",
                "utterance": "What is the largest planet in our solar system?",
                "target_value": "Jupiter",
                "context": {"csv": "", "html": "", "tsv": ""}
            },
            {
                "id": "3",
                "utterance": "Who wrote 'To Kill a Mockingbird'?",
                "target_value": "Harper Lee",
                "context": {"csv": "", "html": "", "tsv": ""}
            },
            {
                "id": "4",
                "utterance": "What is 5 + 7?",
                "target_value": "12",
                "context": {"csv": "", "html": "", "tsv": ""}
            },
            {
                "id": "5",
                "utterance": "What is the chemical symbol for water?",
                "target_value": "H2O",
                "context": {"csv": "", "html": "", "tsv": ""}
            },
            {
                "id": "6",
                "utterance": "Who painted the Mona Lisa?",
                "target_value": "Leonardo da Vinci",
                "context": {"csv": "", "html": "", "tsv": ""}
            },
            {
                "id": "7",
                "utterance": "What is the square root of 81?",
                "target_value": "9",
                "context": {"csv": "", "html": "", "tsv": ""}
            },
            {
                "id": "8",
                "utterance": "What is the speed of light in vacuum (in m/s)?",
                "target_value": "299792458",
                "context": {"csv": "", "html": "", "tsv": ""}
            },
            {
                "id": "9",
                "utterance": "What year did World War II end?",
                "target_value": "1945",
                "context": {"csv": "", "html": "", "tsv": ""}
            },
            {
                "id": "10",
                "utterance": "How many continents are there on Earth?",
                "target_value": "7",
                "context": {"csv": "", "html": "", "tsv": ""}
            },
        ]

        # Store predictions and references for evaluation
        predictions = []
        references = []

        # Send each question to the LLM
        for example in test_data:
            question = example["utterance"]
            prediction = self.agent.eval(question=question).content.strip()
            predictions.append(prediction)
            references.append(example["target_value"])

        # Evaluate with the exact match metric
        print(f"\nPredictions: {predictions}")
        print(f"References: {references}")
        exact_match_result = self.exact_match_metric.compute(predictions=predictions, references=references)

        # Log and assert results
        print(f"Exact Match Score: {exact_match_result['exact_match']}")
        self.assertEqual(exact_match_result["exact_match"], 1)


if __name__ == "__main__":
    unittest.main()
