import streamlit as st
import pandas as pd

from code_agents.smolagent import SmolCodeAgent
from code_agents.serialization_agent import SerializationAgent, TableFormat
from dotenv import load_dotenv
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()
api_base = os.getenv("API_BASE")
api_key = os.getenv("API_KEY")

# === Streamlit UI ===
st.title("AI Data Analyst for Tabular Data")
st.markdown("Upload a CSV file and ask questions about it using natural language.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

agent_type = st.selectbox("Select Agent Type", ["SerializationAgent", "SmolCodeAgent"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data", df.head())

    user_question = st.text_area("Ask a question about your data:", height=100)

    if st.button("Analyze") and user_question:
        with st.spinner("Thinking..."):
            if agent_type == "SerializationAgent":
                data_analysis_agent = SerializationAgent(
                    llm_model="/models/mistral-nemo-12b",
                    temperature=0,
                    max_retries=2,
                    max_tokens=2048,
                    base_url=api_base,
                    api_key=api_key,
                    format_to_convert_to=TableFormat.HTML,
                )

                result = data_analysis_agent.streamlit_eval(df=df, question=user_question)
            else:
                data_analysis_agent = SmolCodeAgent(
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