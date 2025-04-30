from pathlib import Path
from typing import List

from datasets import load_dataset
from tqdm import tqdm

from agent import Agent
from benchmark.evaluator.utils import EvalConfig


class Evaluator:
    """
    Runs evaluation of an LLM agent on one or more datasets.

    This class loads the specified datasets and uses the given agent
    to generate predictions, comparing them to ground-truth answers.

    Args:
        config (EvalConfig): Evaluation configuration specifying datasets and other behavior.
        agent (Agent): The model or agent responsible for generating predictions.

    Attributes:
        predictor (Agent): The evaluation agent.
        datasets (List[DatasetDict]): Loaded datasets ready for evaluation.
    """

    def __init__(self, config: EvalConfig, agent: Agent):
        self.predictor = agent
        self.datasets = []

        if config.force_redownload is True:
            download_mode = "force_redownload"
        else:
            download_mode = "reuse_dataset_if_exists"

        for dataset in config.datasets:
            self.datasets.append(load_dataset(dataset.value, trust_remote_code=True, download_mode=download_mode))

    def evaluate(self):
        """
        Runs evaluation on all loaded datasets.

        Iterates over each test split, uses the predictor agent to generate answers,
        and compares them to the target values.

        Prints:
            - A tqdm loading bar while evaluating
            - Final evaluation results showing prediction vs target
        """
        results = []
        for dataset in self.datasets:
            for example in tqdm(dataset["test"], desc=f"Evaluating examples from {dataset} dataset"):
                path_obj = Path(example["context"]["csv"])
                pred = self.predictor.eval(question=example["utterance"], dataset=path_obj, additional_info=[])
                results.append((pred, example["target_value"]))
                print(pred)

        print("Evaluation results:")
        for pred, example in results:
            print(f"{pred}: {example['target_value']}")
