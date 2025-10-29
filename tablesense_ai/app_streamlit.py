import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
from pandasai import Agent
#from pandasai.llm import ChatOpenAI

from smolagents import CodeAgent, LiteLLMModel

from agent.code_agent.smolagent import SmolCodeAgent
from agent.serialization.serialization_agent import SerializationAgent, TableFormat

from agent.rag.retriever import PostgreSQLHelper, SQLQueryTool, RetrieverTool


# === 1. Load Environment & Configuration ===
load_dotenv()

# AWS
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_REGION_NAME"] = os.getenv("AWS_REGION_NAME")

# PostgreSQL Configuration
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Local LLM API Configuration
api_base = os.getenv("OLLAMA_BASE_URL")
api_key = os.getenv("OLLAMA_API_KEY")


# === 2. Agent Prompts ===
# TODO: Load your prompt templates here form separate files (e.g. .txt .py or .tmpl)
CONTEXT_PROMPT_BASE = """You are a data analyst assistant.

Workflow rules:
- Only if you need more information, retrieve additional documents using the 'retriever' tool.
- For simple look up question, you can diretly answer it if possible.
- As soon as it becomes more complex use sql_query for SELECT queries. This is the only code you are allowed to generate. The retrieved table snippets may not represent the whole table. So here you can refer to the table via the sql tool.

Reason step-by-step and concise.
"""

CONTEXT_FEW_SHOT = '''
Here are some examples. Be aware that your logic might differ!

===== Document 1 =====
| Trial ID | Patient ID | Enrollment Date | Drug Protocol | Age | BMI | Status | Follow-up Weeks |
|:---|:---|:---|:---|---:|:---|:---|---:|
| T-101 | P-301 | 2024-05-10 | Alpha | 65 | 28.5 | Active | 20 |
| T-102 | P-303 | 2024-06-01 | Beta | 71 | 31.9 | Active | 16 |
| T-103 | P-305 | 2024-06-20 | Gamma | 33 | 25.5 | Active | 12 |
Metadata: {'table_id': 'table_c11d2b0e4f', 'row_range': '0-2', ...}

User Query: What is the BMI of patient P-301?

Thought: The user is asking for the BMI of patient P-301. I can directly answer this question by looking at the retrieved documents. 
Code:                                                                                                                                                                                                                                                                 
```python                                                                                                                                                                                                                                                                
final_answer("The BMI of patient P-301 is 28.5.")                                                                                                                                                                                                                          
```

User Query: What is the average BMI of all patients in trials T-102 and T-103?
Thought: I need to calculate the average BMI for multiple trials (T-102 and T-103) across all relevant patients in the database. I must use the sql_query tool with the AVG() function and ensure both "Trial ID" and BMI columns are correctly referenced.

Code:
```python                                                                                                                                                                                                                                                                
query = """
SELECT AVG(BMI)
FROM table_c11d2b0e4f
WHERE "Trial ID" = 'T-102' OR "Trial ID" = 'T-103';
"""
result = sql_query(query)
# Inspect result
print(result)

# if finished, and no additional steps needed use final_answer("Summarize your result and findings here")

```

Now it's your turn:
'''

# === 3. Initialize Tools and Agent (with Caching) ===
# Use st.cache_resource to avoid re-initializing on every script rerun
@st.cache_resource
def get_db_helper():
    return PostgreSQLHelper(db_url=DB_URL)


@st.cache_resource
def get_tools(_db_helper):
    sql_tool = SQLQueryTool(db_helper=_db_helper)
    retriever_tool = RetrieverTool(db_helper=_db_helper)
    return retriever_tool, sql_tool


@st.cache_resource
def get_agent(_retriever_tool, _sql_tool):
    """Initializes and returns the CodeAgent."""
    if not _retriever_tool or not _sql_tool:
        st.error("Tools are not initialized, cannot create agent.")
        return None
    
    try:
        llm_model = LiteLLMModel(
            #model_id="bedrock/mistral.mistral-small-2402-v1:0",
            model_id="ollama/llama3.1:latest",
            max_retries=2,
            max_tokens=4096,
        )
        agent = CodeAgent(
            tools=[_retriever_tool, _sql_tool],
            model=llm_model,
            max_steps=6,
            verbosity_level=2
        )
        return agent
    except Exception as e:
        st.error(f"Failed to create contextual agent: {e}")
        return None

# Load all components
db_helper = get_db_helper()
retriever_tool, sql_tool = get_tools(db_helper)
agent = get_agent(retriever_tool, sql_tool)


# === 4. Streamlit App ===

st.title("Data Analyst Agent")

tab1, tab2 = st.tabs(["Pure Tabular Data Analysis", "Contextualized Data Analysis"])

# --- Tab 1: Pure Tabular Data Analysis ---
# TODO: Make smolCodeAgent and pandasAI Agent work (v3 change to lite llm)
with tab1:
    st.header("Pure Tabular Data Analysis")
    st.info("Upload a CSV file and ask questions about it using natural language.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    agent_type = st.selectbox("Select Agent Type", ["SerializationAgent", "SmolCodeAgent", "PandasAiAgentV3"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Preview of Data", df.head())

            user_question = st.text_area("Ask a question about your data:", height=100, key="tab1_question")

            if st.button("Analyze", key="tab1_button") and user_question:
                with st.spinner("Thinking..."):
                    try:
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

                        elif agent_type == "PandasAiAgentV2":
                            llm = ChatOpenAI(base_url=api_base, model="/models/mistral-nemo-12b", api_key=api_key, temperature=0)
                            data_analysis_agent = Agent(df, config={"llm": llm, "verbose": True,"enable_cache": False})
                            result = data_analysis_agent.chat(user_question)

                        else: # SmolCodeAgent
                            data_analysis_agent = SmolCodeAgent(
                                model_id="openai//models/mistral-nemo-12b",
                                temperature=0,
                                max_retries=2,
                                max_tokens=2048, # SerializationAgent, SmolCodeAgent, TableFormat
                                base_url=api_base,
                                api_key=api_key,
                            )
                            result = data_analysis_agent.invoke(user_question, df)

                        st.write("### Result")
                        if isinstance(result, str) and result.strip().lower().endswith(".png"):
                            st.image(result)
                        else:
                            st.write(result)
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
                        st.error("Please check your Ollama server is running and the model is available.")
        
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")


# --- Tab 2: Contextualized Agent ---
#TODO: Integrate pandasAI as selectable?
with tab2:
    st.header("Contextualized Data Analysis")
    st.info("This is the full-power agent. It first retrieves relevant documents and table schemas, then uses that context to generate an answer (which may or may not involve SQL).")

    context_question = st.text_input("Ask a question about the data:", key="context_question", value="What is the highest, lowest and average Salary (€)?")

    if st.button("Run Contextualized Agent", key="context_run"):
        if not agent:
            st.error("Contextual agent is not initialized. Check app startup errors.")
        elif not context_question:
            st.warning("Please enter a question.")
        else:
            with st.container(border=True):
                st.info(f"**Query:** {context_question}")
                
                # 1. Run retrieval
                with st.spinner("Step 1: Retrieving relevant documents and schema..."):
                    try:
                        retrieval_output = retriever_tool.forward(context_question)
                        with st.expander("See Retrieval Output"):
                            st.text(retrieval_output)
                    except Exception as e:
                        st.error(f"Error during retrieval: {e}")
                        st.stop()
                
                # 2. Construct final prompt
                agent_input = f"{CONTEXT_PROMPT_BASE}\n{CONTEXT_FEW_SHOT}\nRetrieved:\n{retrieval_output}\n\nUser query: {context_question}"
                
                # 3. Run agent
                with st.spinner("Step 2: Contextual Agent is thinking and executing..."):
                    try:
                        agent_output = agent.run(agent_input)
                        st.subheader("Final Answer")
                        st.markdown(agent_output)
                    except Exception as e:
                        st.error(f"Error during agent execution: {e}")
