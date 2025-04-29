import pathlib
import unittest
from evaluate import load
from serialization.serialization_agent import SerializationAgent


class TestLLMExactMatch(unittest.TestCase):
    def setUp(self):
        """
        Set up the LLM agent and exact match metric for evaluation.
        """
        # Configure the agent with the real LLM
        self.agent = SerializationAgent(
            llm_model="/models/mistral-nemo-12b",
            temperature=0.5,
            max_retries=2,
            max_tokens=2048,
            base_url="http://80.151.131.52:9180/v1",
            api_key="THU-I17468S973-Student-24-25-94682Y1315"
        )

        # Load the exact match metric
        self.exact_match_metric = load("exact_match")

    def test_llm_with_exact_match(self):
        """
        Test LLM responses to 10 basic questions using exact match.
        """
        # Define the questions and ground-truth answers
        test_data = [
            {
                "id": "1",
                "utterance": "What is the capital of France?",
                "target_value": "Paris",
                "context": {}
            },
        ]

        # Create a mock CSV file if needed
        dummy_file_path = pathlib.Path("dummy_data.csv")

        # Store predictions and references for evaluation
        predictions = []
        references = []

        # Send each question to the LLM
        for example in test_data:
            question = example["utterance"]

            prediction = self.agent.eval(question=question, dataset=dummy_file_path, additional_info=[])
            predictions.append(prediction)
            references.append(example["target_value"])
            print(f"Question: {question}\nPrediction: {prediction}\nReference: {example['target_value']}\n")

        # Evaluate with the exact match metric
        print(predictions)
        print(references)
        exact_match_result = self.exact_match_metric.compute(predictions=predictions, references=references)

        # Log and assert results
        print(f"Exact Match Score: {exact_match_result['exact_match']}%")
        self.assertEqual(exact_match_result["exact_match"], 1)


if __name__ == "__main__":
    unittest.main()
