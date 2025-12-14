import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
from pandasai import Agent
#from pandasai.llm import ChatOpenAI
from pathlib import Path
import tempfile
import pandasai as pai

from smolagents import CodeAgent, LiteLLMModel
from pandasai_litellm.litellm import LiteLLM

from agent.code_agent.smolagent import SmolCodeAgent
from agent.serialization.serialization_agent import SerializationAgent, TableFormat

from agent.rag.retriever import PostgreSQLHelper, SQLQueryTool, RetrieverTool
from agent.rag.indexer import convert_pdf

# === 1. Load Environment & Configuration ===
load_dotenv()

# PostgreSQL Configuration
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST", "postgres_db")
DB_PORT = os.getenv("DB_PORT", "5432")
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
- For simple look up question, you can diretly answer it either from text or even tables!
- As soon as table calculations becomes more complex use sql_query for SELECT queries. This is the only code you are allowed to generate. The retrieved table snippets may not represent the whole table. So here you can refer to the table via the sql tool.

Reason step-by-step and concise, if you are finished use the final_answer() tool
"""

CONTEXT_FEW_SHOT = '''
Here are some examples. Be aware that your logic might differ!

User Query: What is the BMI of patient P-301?
Retrieved:
===== Document 1 =====
| Trial ID | Patient ID | Enrollment Date | Drug Protocol | Age | BMI | Status | Follow-up Weeks |
|:---|:---|:---|:---|---:|:---|:---|---:|
| T-101 | P-301 | 2024-05-10 | Alpha | 65 | 28.5 | Active | 20 |
| T-102 | P-303 | 2024-06-01 | Beta | 71 | 31.9 | Active | 16 |
| T-103 | P-305 | 2024-06-20 | Gamma | 33 | 25.5 | Active | 12 |
Metadata: {'table_id': 'table_c11d2b0e4f', 'row_range': '0-2', ...}

Thought: The user is asking for the BMI of patient P-301. I can directly answer this question by looking at the retrieved documents. 
Code:                                                                                                                                                                                                                                                                 
```python                                                                                                                                                                                                                                                                
final_answer("The BMI of patient P-301 is 28.5.")                                                                                                                                                                                                                          
```


User Query: What is the average BMI of all patients in trials T-102 and T-103?

Thought: I need to calculate the average BMI for multiple trials (T-102 and T-103) across all relevant patients in the database. I must use the sql_query tool with the AVG() function and ensure both "Trial ID" and BMI columns are correctly referenced.

Code:
```python                                                                                                                                                                                                                                                                
# Step 1: Write a clear query using an alias
query = """
SELECT AVG("BMI") AS average_bmi
FROM table_c11d2b0e4f
WHERE "Trial ID" = 'T-102' OR "Trial ID" = 'T-103';
"""

# Step 2: Execute the query
result = sql_query(query)

# Step 3: Extract the value. Optionaly you can use the print statement to inspect it here.
avg_bmi_value = result[0]['average_bmi']

# Step 4: If you are finished use the final_answer tool
final_answer(f"The average BMI for trials T-102 and T-103 is {avg_bmi_value}.")

```

Now it's your turn:
'''

smolAgentPrompt = f"""
## Instructions
You are acting as an expert data analyst.
Your role is to respond to user questions about a dataset, which is provided to you as a pandas DataFrame.
For each user question, you should:
Analyze and interpret the data in the DataFrame as needed, which may require calculations, aggregations, filtering, or comparisons depending on the user's request.
Therefore generate Code to execute on a pandas DataFrame to perform required analysis.
Use pandas as your primary tool for data manipulation and analysis.
"""

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
            model_id="mistral/mistral-small-2506",
            api_key="p2BQ5k0qBbOrm89s1WioRsDZmM7oPNqC",
            api_base="https://api.mistral.ai/v1",
            max_retries=2,
            max_tokens=4096,
        )
        agent = CodeAgent(
            tools=[_retriever_tool, _sql_tool],
            model=llm_model,
            max_steps=4,
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
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Data", df.head())

        user_question = st.text_area("Ask a question about your data:", height=100, key="tab1_question")

        if st.button("Analyze", key="tab1_button") and user_question:
            with st.spinner("Thinking..."):
                try:
                    if agent_type == "SerializationAgent":
                        # download datase
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = Path(temp_file.name)
                        data_analysis_agent = SerializationAgent(
                            #llm_model="bedrock/mistral.mistral-small-2402-v1:0",
                            llm_model="ollama/llama3.1:latest",
                            temperature=0,
                            max_retries=2,
                            max_tokens=2048,
                            #base_url=api_base,
                            #api_key=api_key,
                            format_to_convert_to=TableFormat.HTML,
                        )
                        result = data_analysis_agent.eval(dataset=temp_file_path, question=user_question, dataset_prompt=None)
                            # 3. Clean up (delete) the temporary file
                            #if temp_file and os.path.exists(temp_file.name):
                            #    os.remove(temp_file.name)

                    elif agent_type == "PandasAiAgentV3":
                        #llm = ChatOpenAI(base_url=api_base, model="/models/mistral-nemo-12b", api_key=api_key, temperature=0)
                        llm = LiteLLM(
                            #model="bedrock/mistral.mistral-small-2402-v1:0",
                            model="ollama/llama3.1:latest",
                            max_retries=2,
                            max_tokens=4096,
                        )
                        pandasAI_agent = Agent(df, config={"llm": llm, "save_logs": True,"verbose": True,"enable_cache": False, "max_retries": 3})
                        result = pandasAI_agent.chat(user_question)

                    else: # SmolCodeAgent
                        smol_model = LiteLLMModel(
                                    model_id="bedrock/mistral.mistral-small-2402-v1:0",
                                    #model_id="ollama/llama3.1:latest",
                                    max_retries=2,
                                    max_tokens=4096,
                                )
                        smolAgent = CodeAgent(
                                    model=smol_model,
                                    tools=[],
                                    additional_authorized_imports=[
                                        "pandas", "numpy", "datetime", "matplotlib", "matplotlib.pyplot",
                                        "plotly", "seaborn", "sklearn", "scikit-learn", "scipy", "plotly.express","statsmodels",
                                        "plotly.graph_objects"
                                    ],
                                    max_steps=10
                                )
                        agent_input = f"{smolAgentPrompt}\n{user_question}"
                        result = smolAgent.run(agent_input, additional_args={"df": df})

                    st.write("### Result")
                    image_path = None

                    if isinstance(result, dict) and result.get('type') == 'plot':
                        image_path = result.get('value')
                        st.image(image_path)
                    elif isinstance(result, str) and result.strip().lower().endswith(".png"):
                        image_path = result.strip()
                        st.image(image_path)
                    else:
                        st.write(result)

                except Exception as e:
                    st.error(f"Error during analysis: {e}")


# --- Tab 2: Contextualized Agent ---
#TODO: Integrate pandasAI as selectable? PandasAIv3 now also generates sql code by default. With SQL connector it can make use of our databse
# === UPLOAD SECTION USING INDEXER ===
# --- Tab 2: Contextualized Agent ---
with tab2:
    st.header("Contextualized Data Analysis")
    
    # === UPLOAD SECTION ===
    # 1. UI Inputs inside the Expander
    with st.expander("📂 Upload Knowledge Base (PDF)", expanded=False):
        st.info("Upload PDF reports. Docling will parse text/tables, storing them in Vector DB and SQL.")
        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], key="context_pdf_uploader")
        
        # We place the button here, but we capture its state
        index_btn = st.button("Index Document", key="index_btn")

    # 2. Processing Logic OUTSIDE the Expander (Fixes the Nesting Error)
    if index_btn and uploaded_pdf is not None:
        # st.status is now at the root level of the tab, not inside the expander
        with st.status("Processing Document...", expanded=True) as status:
            try:
                # 1. Save uploaded file to temp path
                st.write("Saving temporary file...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_pdf.getvalue())
                    tmp_path = Path(tmp_file.name)
                
                # 2. Call the imported indexer function
                st.write("Running Docling & Indexing...")
                
                #  - Conceptual visual for user
                
                convert_pdf(tmp_path)
                
                # 3. Cleanup
                os.remove(tmp_path)
                
                status.update(label="Indexing Complete!", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="Indexing Failed", state="error")
                st.error(f"An error occurred: {e}")
    
    elif index_btn and uploaded_pdf is None:
        st.warning("Please upload a PDF file first.")

    st.divider()

    context_question = st.text_input("Ask a question about the data:", key="context_question", value="What is the highest, lowest and average Salary?")

    if st.button("Run Contextualized Agent", key="context_run"):
        if not agent:
            st.error("Contextual agent is not initialized.")
        elif not context_question:
            st.warning("Please enter a question.")
        else:
            with st.container(border=True):
                st.info(f"**Query:** {context_question}")

                # 1. Retrieval
                with st.spinner("Retrieving relevant context..."):
                    try:
                        retrieval_output = retriever_tool.forward(context_question)
                        with st.expander("See Retrieval Output"):
                            st.text(retrieval_output)
                    except Exception as e:
                        st.error(f"Error during retrieval: {e}")
                        st.stop()
                
                # 2. Prompt & Run
                agent_input = f"{CONTEXT_PROMPT_BASE}\n{CONTEXT_FEW_SHOT}\nUser query: {context_question}\nRetrieved:\n{retrieval_output}"
                
                with st.spinner("Contextual Agent is thinking..."):
                    try:
                        agent_output = agent.run(agent_input)
                        st.subheader("Final Answer")
                        st.markdown(agent_output)
                    except Exception as e:
                        st.error(f"Error during agent execution: {e}")
