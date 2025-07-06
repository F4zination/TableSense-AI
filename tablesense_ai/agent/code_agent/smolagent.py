import os
import pathlib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from smolagents import LiteLLMModel, CodeAgent
from tablesense_ai.agent.base import BaseAgent  # measure_performance
from tablesense_ai.utils.performance import measure_performance

load_dotenv()
base_url = os.getenv("API_BASE")
api_key = os.getenv("API_KEY")
model_id = os.getenv("MODEL_ID")
AWS_REGION = os.getenv("AWS_REGION")

class SmolCodeAgent(BaseAgent):
    def __init__(self, llm_model, temperature, max_retries, max_tokens):
        super().__init__(llm_model, temperature, max_retries, max_tokens, base_url, api_key )
        self.llm_model = LiteLLMModel(
            model_id=model_id,
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens
        )
        self.code_agent = CodeAgent(
            model=self.llm_model,
            tools=[],
            additional_authorized_imports=[
                "pandas", "numpy", "datetime", "matplotlib", "matplotlib.pyplot",
                "plotly", "seaborn", "sklearn", "scikit-learn", "scipy", "plotly.express","statsmodels",
                "plotly.graph_objects"
            ],
            max_steps=5
        )

    # Possible to store the dataframe as csv to provide a path for eval
    def eval(self, question: str, dataset: pathlib.Path, additional_info: str) -> str:
        df = pd.read_csv(dataset, on_bad_lines='skip')
        prompt = additional_info + "\n" + question
        return self.invoke(prompt, df)

    @measure_performance
    def invoke(self, question: str, df: pd.DataFrame) -> str:
        prompt = f"""
## Instructions
You are acting as an expert data analyst.
Your role is to respond to user questions about a dataset, which is provided to you as a pandas DataFrame.
For each user question, you should:
Analyze and interpret the data in the DataFrame as needed, which may require calculations, aggregations, filtering, or comparisons depending on the user's request.
Therefore generate Code to execute on a pandas DataFrame to perform required analysis.
Use pandas as your primary tool for data manipulation and analysis.
Do not provide code, demonstrations, or step-by-step explanations to the user; instead, directly answer the user's question.
Only return the requested answer to the question, nothing more!
Only refer to the data available in the DataFrame when constructing your answer.
If a question requires numerical results (e.g., averages, sums), provide the computed figure within the answer.
Assume each question stands on its own; do not reference previous questions or context beyond the current input.
Your output should always be a single string with no code, comments, or formatting syntax.
Focus on precision and informativeness in your response. Always communicate clearly and avoid unnecessary detail.
Keep your answer AS SHORT AS POSSIBLE.



## User Question:
{question}
"""
        return self.code_agent.run(prompt, additional_args={"df": df})

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
                model_id="mistral.mistral-small-2402-v1:0",
                temperature=0,
                max_retries=2,
                max_tokens=200
            )

            result = data_analysis_agent.invoke(user_question, df)

        st.write("### Result")
        st.write(result)