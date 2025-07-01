import sys

from tablesense_ai.agent.code_agent.smolagent import SmolCodeAgent

sys.path.append("C:/Users/Marco/workspace/TableSenseAI")  # Adjust the path to your project structure

import pickle
from benchmark.evaluator.dataset_definition import SimpleTest, WikiTableQuestions, TabMWP

from benchmark.evaluator.evaluator import Evaluator
from benchmark.evaluator.evaluator import EvalConfig
from tablesense_ai.agent.serialization.serialization_agent import SerializationAgent


# Create your agent instance to test
agent = SerializationAgent(llm_model="mistral.mistral-small-2402-v1:0",
                      temperature=0,
                      max_retries=2,
                      max_tokens=200,
                      base_url="http://Bedroc-Proxy-zVlhZeY8DKqo-1848712918.us-east-1.elb.amazonaws.com/api/v1",
                      api_key="THU-I17468S973-Student-24-25-94682Y1315")


# Configure your evaluation instance
config = EvalConfig([WikiTableQuestions()], False, True, True)
evaluator = Evaluator(config, agent)


# Start the evaluation process
results = evaluator.evaluate()
with open("evaluation_results.pkl", "wb") as f:
    pickle.dump(results, f)