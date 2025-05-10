# TableSense-AI Benchmarking Toolkit

This package provides a framework for benchmarking Large Language Models (LLMs) on table-based question answering tasks. It allows you to define datasets, evaluate LLM agents, and report results using a consistent and extensible interface.

## Quickstart

1. Install dependencies `pip install -r requirements.txt`

2. Define Your Agent: Implement the Agent class in agent.py. It must have an eval() method:

```
class Agent:
    def eval(self, question: str, dataset: Path, additional_info: list) -> str:
        # Call your LLM here and return its prediction
        ...
```

3. Run Evaluation

In your script:

```
from benchmark.evaluator.evaluator import Evaluator
from benchmark.evaluator.evaluator import EvalConfig
from benchmark.evaluator.utils import Dataset

from agent import Agent

agent = Agent()
config = EvalConfig(datasets=[Dataset.SimpleTest], force_redownload=False)

evaluator = Evaluator(config=config, agent=agent)
evaluator.evaluate()
```

## Datasets

### Currently implemented Datasets

- Simple Test Dataset (10 simple examples to test general functionality of your agent): `Dataset.SimpleTest`

### Dataset Format

Each dataset is defined as a script (e.g., simple_test/dataset.py) with examples in JSON format:
```
[
  {
    "id": "nu-1",
    "utterance": "how many people were murdered in 1940/41?",
    "target_value": "100,000",
    "context": {
      "csv": "examples/tables_1/149.csv",
      "html": "examples/tables_1/149.html",
      "tsv": "examples/tables_1/149.tsv"
    }
  }
]
```
These paths are automatically resolved relative to the JSON file.
