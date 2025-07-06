import pickle

from benchmark.evaluator.dataset_definition import FreeformTableQA, WikiTableQuestions
from benchmark.evaluator.evaluator import Evaluator
from benchmark.evaluator.evaluator import EvalConfig
from tablesense_ai.agent.code_agent.smolagent import SmolCodeAgent
from tablesense_ai.agent.serialization.serialization_agent import SerializationAgent


# Create your agent instance to test
agent = SmolCodeAgent(llm_model="mistral.mistral-small-2402-v1:0",
                           temperature=0,
                           max_retries=2,
                           max_tokens=200
                           )


# Configure your evaluation instance
config = EvalConfig([WikiTableQuestions()], True, True)
evaluator = Evaluator(config, agent)


# Start the evaluation process
results = evaluator.evaluate()
with open("evaluation_results.pkl", "wb") as f:
    pickle.dump(results, f)