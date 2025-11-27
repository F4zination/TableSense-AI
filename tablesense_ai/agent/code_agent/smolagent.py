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
You are an expert Python CodeAgent with access to a pandas DataFrame `df`.
Your goal is to return the correct answer string using `final_answer()`.

### STEP 1: INSPECT & CLEAN (Crucial!)
Before solving, run code to check `df.head()` and `df.dtypes`.
1. **Remove Summary Rows:** If the table contains a 'Total' or 'Sum' row at the bottom, DROP it immediately to avoid double counting.
2. **Numeric Conversion:** If a column *looks* numeric (contains digits) but is type object/string:
   - Clean it! Remove characters like '$', '%', ',' using regex.
   - Convert to numeric with `pd.to_numeric(..., errors='coerce')`.
   - *Reason:* Sorting strings like "9" vs "10" yields wrong results ("9" > "10"). Math on strings like "5"*3 yields "555".

### STEP 2: SOLVE BASED ON QUESTION TYPE

**TYPE A: CALCULATIONS (How many, Sum, Average, Difference, Range)**
- **Rate of Change:** Calculate as simple difference (`Value2 - Value1`), NOT division/percentage.
- **Range:** Calculate as `Max - Min`.
- **Aggregations:** Ignore `NaN` values.
- **Lists:** Only use `explode` if the cell contains a list AND the question implies counting individual items.

**TYPE B: LOOKUP / RETRIEVAL (Who, Which, First, Highest, Lowest)**
- **"Highest/Lowest":** Use `idxmax()` or `nlargest()` on the CLEANED numeric column, then return the label/name from the target column.
- **Text Filters:** Use `.str.contains(..., case=False)` for robust matching.
- **Format:** Return the exact string as found in the dataframe (unless formatting is broken).

### STEP 3: FINAL OUTPUT
- Re-attach magnitudes (billions, years) to the answer if they were removed during cleaning.
- Return **ONLY** the result value in `final_answer()`. No explanations.

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
