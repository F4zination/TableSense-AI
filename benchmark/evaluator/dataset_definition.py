from abc import ABC

from benchmark.evaluator.metrics.exact_match_metric import ExactMatchMetric
from benchmark.evaluator.metrics.bertscore_metric import BERTScoreMetric
from benchmark.evaluator.metrics.rogue_metric import RogueMetric
from benchmark.evaluator.metrics.metric import Metric


class Dataset(ABC):
    """
    Enum for available datasets.

    Attributes:
        SimpleTest (str): Path to the simple test dataset script.
    """

    def __init__(self, dataset_path: str, is_remote: bool, metric: list[Metric], system_prompt):
        self.dataset_path = dataset_path
        self.is_remote = is_remote
        self.metrics = metric
        self.system_prompt = system_prompt


class SimpleTest(Dataset):
    def __init__(self):
        system_prompt = ""
        super().__init__(dataset_path="tab_llm_datasets/simple_test/dataset.py", is_remote=False,
                         metric=[ExactMatchMetric(), BERTScoreMetric(), RogueMetric()], system_prompt=system_prompt)


class WikiTableQuestions(Dataset):
    def __init__(self):
        system_prompt = "Keep the answer as short as possible!!!\n"
        super().__init__(dataset_path="TableSenseAI/WikiTableQuestions", is_remote=True,
                         metric=[ExactMatchMetric(),RogueMetric()], system_prompt=system_prompt)


class FreeformTableQA(Dataset):
    def __init__(self):
        system_prompt = ""
        super().__init__(dataset_path="TableSenseAI/FreeformTableQA", is_remote=True,
                         metric=[ExactMatchMetric(), BERTScoreMetric(), RogueMetric()], system_prompt=system_prompt)


class TabMWP(Dataset):
    def __init__(self):
        system_prompt = "Keep the answer as short as possible!!!\n"

        super().__init__(dataset_path="TableSenseAI/TabMWP", is_remote=True,
                         metric=[ExactMatchMetric(),RogueMetric()], system_prompt=system_prompt)


class FreeformTableQASelection(Dataset):
    def __init__(self):
        system_prompt = """Answer only in one setence!!!"""
        super().__init__(dataset_path="TableSenseAI/FreeformTableQASelection", is_remote=True,
                         metric=[ExactMatchMetric(), BERTScoreMetric(), RogueMetric()], system_prompt=system_prompt)


class TabMWPSelection(Dataset):
    def __init__(self):
        system_prompt = """Keep the answer as short as possible!!!
Follow these formatting rules:
- Do not use thousand separators (1000, not 1,000).  
- Use "." for decimals (10.45, not 10,45).  
- Round numbers to 2 decimal places (10.45, not 10.45321).  
- Do not add decimals to whole numbers (10, not 10.00).  
- Return fractions in their simplest form, not as decimals.  
- Remove leading currency symbols ($1000 → 1000).  
- When the answer is a text value that must be inferred or generated (not copied from the table), return it in lowercase. 
- Match the table’s existing formatting style.  
- Output only the final answer value, never show the calculation.
"""
        super().__init__(dataset_path="TableSenseAI/TabMWPSelection", is_remote=True,
                          metric=[ExactMatchMetric(), RogueMetric()], system_prompt=system_prompt)


class WikiTableQuestionsSelection(Dataset):
    def __init__(self):
        system_prompt = """Keep the answer as short as possible!!!
Follow these formatting rules:
- Do not use thousand separators (1000, not 1,000).  
- Use "." for decimals (10.45, not 10,45).  
- Round numbers to 2 decimal places (10.45, not 10.45321).  
- Use | as seperator for multiple year answers (e.g., 1999|2000|2001 , NOT 1999, 2000, 2001).
- If the answer is surrounded by quotes, remove them (e.g., "New York" -> New York).
"""
        super().__init__(dataset_path="TableSenseAI/WikiTableQuestionsSelection", is_remote=True,
                          metric=[ExactMatchMetric(), RogueMetric()], system_prompt=system_prompt)
