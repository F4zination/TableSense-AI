from evaluate import load

from benchmark.evaluator.dataset_definition import SimpleTest, WikiTableQuestions
from benchmark.evaluator.evaluator import Evaluator
from benchmark.evaluator.evaluator import EvalConfig
<<<<<<< HEAD
from benchmark.evaluator.metrics.exact_match_metric import ExactMatchMetric
from serialization.serialization_agent import SerializationAgent
=======
from benchmark.evaluator.utils import Dataset
from tablesense_ai.agent.serialization.serialization_agent import SerializationAgent
>>>>>>> 09ff525 (refactor: reorganize project into a Python package)

# Create your agent instance to test
agent = SerializationAgent(llm_model="/models/mistral-nemo-12b",
                           temperature=0,
                           max_retries=2,
                           max_tokens=2048,
                           base_url="http://80.151.131.52:9180/v1",
                           api_key="THU-I17468S973-Student-24-25-94682Y1315")

# Configure your evaluation instance
config = EvalConfig([SimpleTest()], [ExactMatchMetric()], False)
evaluator = Evaluator(config, agent)

# Start the evaluation process
evaluator.evaluate()
