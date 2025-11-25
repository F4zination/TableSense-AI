import os
import pickle

from benchmark.evaluator.dataset_definition import FreeformTableQA, WikiTableQuestions, TabMWP, SimpleTest, \
    TabMWPSelection, FreeformTableQASelection, WikiTableQuestionsSelection
from benchmark.evaluator.evaluator import Evaluator
from benchmark.evaluator.evaluator import EvalConfig
from tablesense_ai.agent.code_agent.smolagent import SmolCodeAgent
from tablesense_ai.agent.serialization.serialization_agent import SerializationAgent, TableFormat


"""system_prompt_serialization = (
    "You are an expert data analyst.\n\n"
    "You receive one table.\n"
    "For each user question:\n\n"
    "Base your answer exclusively on the provided table.\n"
    "Output ONLY the final answer string – no labels, no quotes, no code, no explanations.\n"
    "Copy values EXACTLY as stored (case, accents, punctuation).\n"
    "If a cell already includes alternatives (e.g., \"X or Y\"), reproduce it unchanged.\n"
    "Provide numeric values exactly as stored; do NOT add or remove separators or decimals.\n"
    "Preserve original date/time formats verbatim.\n"
    "If the answer is missing or cannot be inferred, output the single token \"N/A\".\n"
    "Return nothing except this answer value.\n"
    "The table is:\n{data}\n"
)"""


formats = {
    "json": TableFormat.JSON,
    "natural": TableFormat.NATURAL,
    "html": TableFormat.HTML,
    "csv": TableFormat.CSV,
    "md": TableFormat.MARKDOWN
}

# for fmt_key, fmt_value in formats.items():
#     print(f"Running evaluation for format: {fmt_key}")
    # agent = SerializationAgent(
    #     llm_model="mistral-small-2506",
    #     temperature=0,
    #     max_retries=2,
    #     max_tokens=100,
    #     base_url="https://api.mistral.ai/v1",
    #     api_key = os.getenv("OPENAI_API_KEY"),
    #     format_to_convert_to=fmt_value,
    #     verbose=True
    # )

agent = SmolCodeAgent(
    llm_model="mistral/mistral-small-2506",
    temperature=0,
    max_retries=5,
    max_tokens=10000,
)


# Configure your evaluation instance
config = EvalConfig([TabMWPSelection()], True, False)
evaluator = Evaluator(config, agent)

# Start the evaluation process
results = evaluator.evaluate()

print(results)

with open(f"evaluation_results.pkl", "wb") as f:
    pickle.dump(results, f)