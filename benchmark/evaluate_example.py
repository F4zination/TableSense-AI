import sys
sys.path.append("C:/Users/Marco/workspace/TableSenseAI")  # Adjust the path to your project structure

import pickle
from benchmark.evaluator.dataset_definition import SimpleTest, WikiTableQuestions, TabMWP

from benchmark.evaluator.evaluator import Evaluator
from benchmark.evaluator.evaluator import EvalConfig
from tablesense_ai.agent.serialization.serialization_agent import SerializationAgent


# Create your agent instance to test
agent = SerializationAgent(llm_model="/models/mistral-nemo-12b",
                           temperature=0,
                           max_retries=2,
                           max_tokens=200,
                           base_url="http://80.151.131.52:9180/v1",
                           api_key="THU-I17468S973-Student-24-25-94682Y1315", verbose=True)


# Configure your evaluation instance
config = EvalConfig([WikiTableQuestions()], False, True, True)
evaluator = Evaluator(config, agent)


# Start the evaluation process
results = evaluator.evaluate()
with open("evaluation_results.pkl", "wb") as f:
    pickle.dump(results, f)