from agent.serialization.converter import TableFormat
from agent.serialization.serialization_agent import SerializationAgent
import pathlib


def main():
    # Initialize the SerializationAgent with the required parameters
    agent = SerializationAgent(
        llm_model="/models/mistral-nemo-12b",
        temperature=0,
        max_retries=2,
        max_tokens=2048,
        base_url="http://80.151.131.52:9180/v1",
        api_key="THU-I17468S973-Student-24-25-94682Y1315",
        format_to_convert_to=TableFormat.HTML
    )

    # Example usage of the eval method
    question = "What is Peters Email?"
    dataset = pathlib.Path("input/minimal.csv")


    response = agent.eval(question, dataset, None)
    print(f"Response: {response}")


if __name__ == "__main__":
    main()