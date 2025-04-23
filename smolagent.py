import os
import pandas as pd
import smolagents
from smolagents import LiteLLMModel, CodeAgent
import litellm

litellm._turn_on_debug()

## setting up LLM Model
llm_model = LiteLLMModel(
    #model="models/mistral-nemo-12b",
    model_id="openai/models/mistral-nemo-12b",
    temperature=0,
    max_retries=2,
    max_tokens=2048,
    api_base="http://80.151.131.52:9180/v1",
    api_key="THU-I17468S973-Student-24-25-94682Y1315"
)


system_prompt = """
## Instructions
You are acting as an expert data analyst.
Your job is to answer questions asked by the user about the given dataset.

## Data information
{df_head}

## User Question:
{query}
"""

agent = CodeAgent(
    tools=[],  # only use built-in tools
    model=llm_model,
    add_base_tools=True,
    additional_authorized_imports=[
        "pandas", "numpy", "datetime",
        "matplotlib", "plotly", "seaborn", "sklearn"
    ]
)


def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df


def main():
    df = load_csv("titanic.csv")

    user_question = "How many male and female survived?"
    context = system_prompt.format(df_head=df.head(3).to_string(), query=user_question)

    result = agent.run(context)
    print(result)


if __name__ == "__main__":
    main()
