import pandas as pd
import streamlit as st
from smolagents import LiteLLMModel, CodeAgent
from dotenv import load_dotenv
import os

load_dotenv()
api_base = os.getenv("API_BASE")
api_key = os.getenv("API_KEY")

# Define LLM model
llm_model = LiteLLMModel(
    model_id="openai//models/mistral-nemo-12b",
    temperature=0,
    max_retries=2,
    api_base=api_base,
    api_key=api_key,
)


# Option to provide here data description as df.dtypes. At the moment dataframe 
prompt_template= """
## Instructions
You are acting as an expert data analyst.
Your job is to answer question asked by the user about the dataset, provided as a pandas DataFrame. 
Therefore generate Code to execute on a pandas DataFrame.

## User Question:
{query}
"""

# Initialize CodeAgent
agent = CodeAgent(
    tools=[],
    model=llm_model,
    additional_authorized_imports=[
        "pandas", "numpy", "datetime",
        "matplotlib", "matplotlib.pyplot", "plotly",
        "seaborn", "sklearn", "scikit-learn", "scipy"
    ]
)


# Streamlit UI
st.title("AI Data Analyst")
st.markdown("Upload a CSV file and ask questions about it using natural language.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data", df.head())

    user_question = st.text_area("Ask a question about your data:", height=100)

    if st.button("Analyze") and user_question:
        with st.spinner("Thinking..."):
            prompt = prompt_template.format(query=user_question, df_dtypes=df.dtypes)

            # Instead of df, you can directly prompt the csv file. The LLM will convert it into a df
            result = agent.run(prompt, additional_args={"dataframe": df})

        st.write("### Result")
        st.write(result)
