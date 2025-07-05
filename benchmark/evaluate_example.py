import pickle

from benchmark.evaluator import helper
from benchmark.evaluator.dataset_definition import FreeformTableQA, WikiTableQuestions, SimpleTest
from benchmark.evaluator.evaluator import Evaluator
from benchmark.evaluator.evaluator import EvalConfig
from tablesense_ai.agent.serialization.serialization_agent import SerializationAgent


iterations = 2

# Create your agent instance to test
agent = SerializationAgent(llm_model="/models/mistral-nemo-12b",
                           temperature=0,
                           max_retries=2,
                           max_tokens=200,
                           base_url="http://80.151.131.52:9180/v1",
                           api_key="THU-I17468S973-Student-24-25-94682Y1315", verbose=True)

complete_results = []

for i in range(iterations):
    # Configure your evaluation instance
    config = EvalConfig([SimpleTest(), WikiTableQuestions()], True, True)
    evaluator = Evaluator(config, agent)

    # Start the evaluation process
    results = evaluator.evaluate()

    print(results)
    complete_results.append(results)


averaged = helper.average_results(complete_results)
print(averaged)


with open("evaluation_results.pkl", "wb") as f:
    pickle.dump(complete_results, f)
