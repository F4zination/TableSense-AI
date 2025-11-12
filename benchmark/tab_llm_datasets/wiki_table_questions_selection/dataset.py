import os

from datasets import DatasetInfo, GeneratorBasedBuilder, SplitGenerator, Split, Features, Value


class WikiTableQuestionsSelection(GeneratorBasedBuilder):
    """
    A simple Hugging Face dataset builder for evaluating question-answering (QA)
    over tabular data, using file paths as context (CSV, HTML, TSV).

    The dataset is loaded from a JSON file containing QA samples and context file paths.
    """

    def _info(self):
        """
        Returns the metadata and schema of the dataset.

        Returns:
            DatasetInfo: Contains description, features (schema), and supervised keys.
        """
        return DatasetInfo(
            description="QA over tabular data with file paths as context",
            features=Features({
                "id": Value("string"),
                "utterance": Value("string"),
                "target_value": Value("string"),
                "context": Features({
                    "csv": Value("string"),
                    "html": Value("string"),
                    "tsv": Value("string"),
                }),
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """
        Downloads and defines dataset splits.

        Args:
            dl_manager (DownloadManager): The Hugging Face datasets download manager.

        Returns:
            List[SplitGenerator]: A list containing a single test split generator.
        """
        data_path = dl_manager.download("examples/examples-test.json")
        return [
            SplitGenerator(name=Split.TEST, gen_kwargs={"filepath": data_path}),
        ]

    def _generate_examples(self, filepath):
        """
        Yields examples from the dataset JSON file.

        Each example consists of a question, target value, and paths to context files
        (CSV, HTML, TSV). The relative paths are resolved into absolute paths based
        on the JSON file's directory.

        Args:
            filepath (str): Path to the JSON file containing dataset examples.

        Yields:
            Tuple[int, dict]: A tuple of the index and the data sample dictionary.
        """
        import json
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        for i, item in enumerate(data):
            yield i, {
                "id": item["id"],
                "utterance": item["utterance"],
                "target_value": item["target_value"],
                "context": {
                    "csv": item["context"]["csv"],
                    "html": item["context"]["html"],
                    "tsv": item["context"]["tsv"],
                },
            }
