import os
import pathlib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from smolagents import LiteLLMModel, CodeAgent
from tablesense_ai.agent.base import BaseAgent #measure_performance
from tablesense_ai.utils.performance import measure_performance


load_dotenv()
api_base = os.getenv("API_BASE")
api_key = os.getenv("API_KEY")


class SmolCodeAgent(BaseAgent):
    def __init__(self, model_id, temperature, max_retries, max_tokens, base_url, api_key):
        super().__init__(model_id, temperature, max_retries, max_tokens, base_url, api_key)
        self.code_agent = CodeAgent(
            tools=[],
            model=LiteLLMModel(
                model_id=model_id,
                temperature=temperature,
                max_retries=max_retries,
                api_base=base_url,
                api_key=api_key,
            ),
            additional_authorized_imports=[
                "pandas", "numpy", "datetime", "matplotlib", "matplotlib.pyplot",
                "plotly", "seaborn", "sklearn", "scikit-learn", "scipy", "plotly.express","statsmodels",
                "plotly.graph_objects"
            ]
        )

    # Possible to store the dataframe as csv to provide a path for eval
    def eval(self, question: str, dataset: pathlib.Path, additional_info: list[dict]) -> str:
        df = pd.read_csv(dataset)
        return self.invoke(question, df)

    @measure_performance
    def invoke(self, question: str, df: pd.DataFrame) -> str:
        prompt = f"""
## Instructions
You are acting as an expert data analyst.
Your job is to answer question asked by the user about the dataset, provided as a pandas DataFrame.
Therefore generate Code to execute on a pandas DataFrame.

## User Question:
{question}
"""
        return self.code_agent.run(prompt, additional_args={"df": df})


if __name__ == "__main__":
    # === Streamlit UI ===
    st.title("AI Data Analyst for Tabular Data")
    st.markdown("Upload a CSV file and ask questions about it using natural language.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Data", df.head())

        user_question = st.text_area("Ask a question about your data:", height=100)

        if st.button("Analyze") and user_question:
            with st.spinner("Thinking..."):
                data_analysis_agent= SmolCodeAgent(
                    model_id="openai//models/mistral-nemo-12b",
                    temperature=0,
                    max_retries=2,
                    max_tokens=2048,
                    base_url=api_base,
                    api_key=api_key,
                )
        
                result = data_analysis_agent.invoke(user_question, df)

            st.write("### Result")
            st.write(result)