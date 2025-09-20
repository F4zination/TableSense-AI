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
        system_prompt = (
            "WikiTableQuestions (WTQ) — dataset-specific notes:\n"
            "- Answers are usually single cells or simple aggregations.\n"
            "- If multiple distinct cells are required, join them with '|' (no spaces).\n"
            "- If appropriate, answer strictly 'yes' or 'no'.\n"
            "- Include units if present in the cell (do not invent units).\n"
            "- Reproduce ranges exactly as written (e.g., '1982-1985').\n"
            "- Do not include calculation steps or equations; output only the final value (e.g., '591', not '129+91+68+90+62+41+33+39+76+62=591').\n"
            "- Keep the answer as short as possible.\n"
            "\n"
            "Examples (from WTQ test set):\n"
            "1) Q: how many people were murdered in 1940/41?\n"
            "   A: 100,000\n"
            "\n"
            "2) Q: what time period had no shirt sponsor?\n"
            "   A: 1982-1985\n"
        )
        super().__init__(dataset_path="TableSenseAI/WikiTableQuestions", is_remote=True,
                         metric=[ExactMatchMetric(),RogueMetric()], system_prompt=system_prompt)


class FreeformTableQA(Dataset):
    def __init__(self):
        system_prompt = (
            "FreeformTableQA — dataset-specific notes:\n"
            "- Style: Answer in 1–2 concise, well-formed sentences.\n"
            "- When listing or comparing items, summarize naturally in prose (avoid '|' joins).\n"
            "- Preserve original range punctuation (e.g., '2012–13') and include units if present.\n"
            "\n"
            "Examples (from FreeformTableQA test set):\n"
            "1) Q: What TV shows was Shagun Sharma seen in 2019?\n"
            "   A: In 2019, Shagun Sharma played in the roles as Pernia in Laal Ishq, Vikram Betaal Ki Rahasya Gatha as Rukmani/Kashi and Shaadi Ke Siyape as Dua.\n"
            "\n"
            "2) Q: How much overall damage did the German submarine U-438 cause?\n"
            "   A: The U-438 sank three ships, totalling 12,045 gross register tons (GRT) and damaged one ship totalling 5,496 GRT.\n"
        )
        super().__init__(dataset_path="TableSenseAI/FreeformTableQA", is_remote=True,
                         metric=[ExactMatchMetric(), BERTScoreMetric(), RogueMetric()], system_prompt=system_prompt)


class TabMWP(Dataset):
    def __init__(self):
        system_prompt = (
            "TabMWP — dataset-specific notes:\n"
            "- Compute the single final result required by the question using the table.\n"
            "- Return exactly one value; do not include formulas, intermediate steps, or multiple values (no '|' joins).\n"
            "- Perform necessary arithmetic (add, subtract, multiply, divide); do not round unless the question explicitly asks you to.\n"
            "- If the question is yes/no, answer strictly 'yes' or 'no'.\n"
            "- If units/symbols (e.g., $, %, p.m.) appear in the table or question, include them and match their style and separators (e.g., '$7,706', '12:35 p.m.').\n"
            "- Use a minus sign for negative values when appropriate (e.g., '-2').\n"
        )

        super().__init__(dataset_path="TableSenseAI/TabMWP", is_remote=True,
                         metric=[ExactMatchMetric(),RogueMetric()], system_prompt=system_prompt)
