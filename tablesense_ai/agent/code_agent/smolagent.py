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
            model_id=llm_model,
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens
        )
        self.code_agent = CodeAgent(
            model=self.llm_model,
            tools=[],
            additional_authorized_imports=[
                "pandas", "numpy", "datetime", "matplotlib", "matplotlib.pyplot",
                "plotly", "seaborn", "sklearn", "scikit-learn", "scipy", 
                "plotly.express","statsmodels", "plotly.graph_objects",
                "fractions", "re", "math" 
            ],
            #Expand to 15 for better results
            max_steps=10
        )

    # Possible to store the dataframe as csv to provide a path for eval
    def eval(self, question: str, dataset: pathlib.Path, dataset_prompt: str) -> str:
        df = pd.read_csv(dataset, on_bad_lines='skip')
        prompt = dataset_prompt + "\n" + question
        return self.invoke(prompt, df)

    @measure_performance
    def invoke(self, question: str, df: pd.DataFrame) -> str:
        prompt = prompt = f"""
You are an expert Python Data Analyst. You are given a pandas DataFrame `df` and a user question.
Your goal is to write Python code that calculates the correct answer and returns it using `final_answer()`.

### CRITICAL: DATA PREPARATION (Execute these steps first)
Real-world data is often dirty. Before answering, you MUST standardize the dataframe:
1.  **Inspect & Clean Headers:** Ensure column names are stripped of whitespace.
2.  **Handle "Total" Rows:** Check `df.tail()` or the first column for rows labeled "Total", "Sum", or "Average". DROP these rows immediately to prevent double-counting in aggregations.
3.  **Fix Data Types:** Columns that look numeric but are type `object` (strings) MUST be cleaned.
    - Remove currency symbols ('$'), commas (','), and percentages ('%').
    - Use `pd.to_numeric(..., errors='coerce')` to convert them to float/int.
    - *Why?* Sorting strings ("9" > "10") or multiplying strings ("5"*3 = "555") leads to wrong answers.

### LOGIC GUIDELINES
- **Question Type "Who/Which":** If asked "Who has the most...", identify the row with the maximum value, but return the **Entity Name** (from the main label column), NOT the value itself.
- **Question Type "Rate of Change":** Unless specified as a percentage, calculate the simple difference: `Value2 - Value1`.
- **Question Type "Range":** Calculate as `Max - Min`.
- **Time/Dates:** If columns contain years (e.g., "1990/91"), extract the main year as an integer for calculations.

### OUTPUT FORMAT
- **Final Answer:** Use `final_answer(result)`.
- **Format:** The result must be a clean string or number. 
- **Units:** If the original column had a unit (e.g., "$", "kg", "years"), append it to the final answer string if appropriate.

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
                llm_model="mistral.mistral-small-2402-v1:0",
                temperature=0,
                max_retries=2,
                max_tokens=200
            )

            result = data_analysis_agent.invoke(user_question, df)

        st.write("### Result")
        st.write(result)
