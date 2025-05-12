from evaluate import load

from benchmark.evaluator.dataset_definition import SimpleTest
from benchmark.evaluator.evaluator import Evaluator
from benchmark.evaluator.evaluator import EvalConfig
from serialization.serialization_agent import SerializationAgent

# Create your agent instance to test
agent = SerializationAgent(llm_model="/models/mistral-nemo-12b",
                           temperature=0,
                           max_retries=2,
                           max_tokens=2048,
                           base_url="http://80.151.131.52:9180/v1",
                           api_key="THU-I17468S973-Student-24-25-94682Y1315")
# Load Exact Match Metrik
exact_match = load("exact_match")


# Configure your evaluation instance
config = EvalConfig([SimpleTest()], False, [exact_match]
)
evaluator = Evaluator(config, agent)

# Start the evaluation process
evaluator.evaluate()
