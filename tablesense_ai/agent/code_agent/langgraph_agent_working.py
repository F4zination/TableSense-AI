import os
import pathlib
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional, Annotated, TypedDict, Sequence
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from datetime import datetime
from io import StringIO # OPTIMIERUNG 2: Hinzugefügt für pd.read_json

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.chat_models import ChatLiteLLM

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from pydantic import BaseModel
from tablesense_ai.agent.base import BaseAgent
from tablesense_ai.utils.performance import measure_performance
import time

load_dotenv()

# ============= Configuration =============
base_url = os.getenv("API_BASE")
api_key = os.getenv("API_KEY")
model_id = os.getenv("MODEL_ID")
AWS_REGION = os.getenv("AWS_REGION")

# ============= Performance Monitoring (unverändert) =============
def langgraph_measure_performance(func):
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            print(f"LangGraph Agent executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"LangGraph Agent failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    return wrapper

class MockMonitor:
    def get_total_token_counts(self):
        return {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

# ============= Data Models (unverändert) =============
class DataProfile(BaseModel):
    shape: tuple
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    memory_usage: float
    sample_data: Optional[List[Dict[str, Any]]] = None

class AnalysisResult(BaseModel):
    answer: str
    confidence: float
    execution_time: float
    approach_used: str
    intermediate_results: Optional[Dict] = None
    visualizations: Optional[List[Dict]] = None

# ============= State Definition (unverändert) =============
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    dataframe_json: Optional[str]
    data_profile: Optional[DataProfile]
    query: str
    intent: Optional[str]
    analysis_result: Optional[AnalysisResult]
    error: Optional[str]
    retry_count: int
    execution_history: List[Dict[str, Any]]
    cache: Dict[str, Any]
    metadata: Dict[str, Any]

# ============= Tools Definition (mit StringIO-Fix) =============

@tool
def profile_dataframe(df_json: str) -> Dict:
    """Profile a pandas DataFrame to understand its structure and content"""
    # OPTIMIERUNG 2: StringIO verwenden
    df = pd.read_json(StringIO(df_json)) 
    
    profile = {
        "shape": df.shape, "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
        "memory_usage": df.memory_usage(deep=True).sum() / 1024**2, # MB
        "sample_data": df.head(3).to_dict(orient='records')
    }
    if profile["numeric_columns"]:
        profile["numeric_summary"] = df[profile["numeric_columns"]].describe().to_dict()
    return profile

@tool
def execute_pandas_code(code: str, df_json: str) -> str:
    """
    Execute pandas code on a DataFrame and return the result as a string.
    The DataFrame is available as 'df'.
    The final result MUST be stored in a variable named 'result'.
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import json
    import sys
    import io
    
    try:
        # OPTIMIERUNG 2: StringIO verwenden
        df = pd.read_json(StringIO(df_json))
        
        safe_globals = {
            'pd': pd, 'pandas': pd, 'np': np, 'numpy': np, 'datetime': datetime,
            'json': json, 'len': len, 'str': str, 'int': int, 'float': float,
            'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
            'range': range, 'enumerate': enumerate, 'zip': zip,
            'sum': sum, 'min': min, 'max': max, 'abs': abs,
            'round': round, 'sorted': sorted, 'print': print,
            '__builtins__': {
                '__import__': __import__,
                'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
                'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                'range': range, 'enumerate': enumerate, 'zip': zip,
                'sum': sum, 'min': min, 'max': max, 'abs': abs,
                'round': round, 'sorted': sorted, 'print': print,
            }
        }
        safe_globals['df'] = df
        
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            exec(code, safe_globals)
            printed_output = captured_output.getvalue()
            
            if 'result' in safe_globals:
                result = safe_globals['result']
                
                # Format the result to be LLM-readable
                if isinstance(result, pd.DataFrame):
                    result_str = result.to_string(max_rows=10, max_cols=10)
                    if len(result) > 10:
                        result_str += f"\n... (showing first 10 of {len(result)} rows)"
                elif isinstance(result, pd.Series):
                    result_str = result.to_string(max_rows=10)
                    if len(result) > 10:
                         result_str += f"\n... (showing first 10 of {len(result)} values)"
                elif hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
                    if len(result) > 20:
                        result_str = f"Collection with {len(result)} items: {list(result)[:10]}... (truncated)"
                    else:
                        result_str = str(result)
                else:
                    result_str = str(result)
                
                return f"Printed Output:\n{printed_output}\nResult:\n{result_str}"
            else:
                if printed_output:
                    return f"Printed Output:\n{printed_output}"
                else:
                    raise NameError("Code executed, but no 'result' variable was defined.")
                    
        finally:
            sys.stdout = old_stdout
            
    except Exception as e:
        raise e

@tool
def generate_visualization(df_json: str, chart_type: str, config: str) -> Dict:
    """Generate a visualization from the DataFrame"""
    import plotly.express as px
    import plotly.graph_objects as go
    import json
    
    # OPTIMIERUNG 2: StringIO verwenden
    df = pd.read_json(StringIO(df_json))
    config_dict = json.loads(config)
    
    try:
        # ... (Rest der Funktion unverändert)
        if chart_type == "scatter":
            fig = px.scatter(df, **config_dict)
        elif chart_type == "bar":
            fig = px.bar(df, **config_dict)
        elif chart_type == "line":
            fig = px.line(df, **config_dict)
        elif chart_type == "histogram":
            fig = px.histogram(df, **config_dict)
        elif chart_type == "box":
            fig = px.box(df, **config_dict)
        elif chart_type == "heatmap":
            fig = px.imshow(df.corr() if not config_dict else df, **config_dict)
        else:
            return {"error": f"Unsupported chart type: {chart_type}"}
        
        return {
            "chart_json": fig.to_json(),
            "chart_type": chart_type,
            "config": config_dict
        }
    except Exception as e:
        return {"error": str(e)}

# ============= LangGraph Agent Implementation (optimiert) =============

class LangGraphTabularAgent(BaseAgent):
    def __init__(self, llm_model, temperature=0, max_retries=3, max_tokens=4096):
        super().__init__(llm_model, temperature, max_retries, max_tokens, base_url, api_key)
        
        self.code_agent = type('MockCodeAgent', (), {'monitor': MockMonitor()})()
        
        # OPTIMIERUNG 3: Korrigierte Initialisierung
        self.llm = ChatLiteLLM(
            model=model_id,
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens,
            api_base=base_url,
            api_key=api_key
        )
        
        self.MAX_CODE_ATTEMPTS = 5
        self.graph = self._build_graph()
        self.current_dataframe = None
        self.app = self.graph.compile()
        
    def _get_dataframe(self, state: AgentState) -> pd.DataFrame:
        if hasattr(self, 'current_dataframe') and self.current_dataframe is not None:
            return self.current_dataframe
        if state.get("dataframe_json"):
            try:
                # OPTIMIERUNG 2: StringIO verwenden
                return pd.read_json(StringIO(state["dataframe_json"]))
            except Exception:
                pass
        return None
        
    def _build_graph(self) -> StateGraph:
        """Build the optimized LangGraph workflow (unverändert)"""
        workflow = StateGraph(AgentState)
        workflow.add_node("profile_data", self.profile_data_node)
        workflow.add_node("classify_intent", self.classify_intent_node)
        workflow.add_node("execute_analysis", self.execute_analysis_node)
        workflow.add_node("execute_visualization", self.execute_visualization_node)
        workflow.add_node("execute_profile_qa", self.execute_profile_qa_node)
        workflow.add_node("synthesize_result", self.synthesize_result_node)
        workflow.add_node("handle_error", self.handle_error_node)
        
        workflow.add_edge(START, "profile_data")
        workflow.add_edge("profile_data", "classify_intent")
        
        workflow.add_conditional_edges(
            "classify_intent",
            self.route_by_intent,
            {
                "analysis": "execute_analysis",
                "visualization": "execute_visualization",
                "profile_qa": "execute_profile_qa",
                "error": "handle_error"
            }
        )
        workflow.add_edge("execute_analysis", "synthesize_result")
        workflow.add_edge("execute_visualization", "synthesize_result")
        workflow.add_edge("execute_profile_qa", "synthesize_result")
        workflow.add_conditional_edges(
            "handle_error",
            self.should_retry,
            {"retry": "classify_intent", "end": END}
        )
        workflow.add_edge("synthesize_result", END)
        return workflow
    
    def profile_data_node(self, state: AgentState) -> AgentState:
        """Profile the DataFrame to understand its structure (unverändert)"""
        df = self._get_dataframe(state)
        query = state["query"]
        state["messages"].append(HumanMessage(content=query))
        
        if df is not None and isinstance(df, pd.DataFrame):
            try:
                profile_dict = profile_dataframe.invoke({"df_json": df.to_json()})
                profile_dict_cleaned = profile_dict.copy()
                profile_dict_cleaned.pop("numeric_summary", None)
                state["data_profile"] = DataProfile(**profile_dict_cleaned)
                state["metadata"]["profiling_complete"] = True
            except Exception as e:
                state["error"] = f"Failed to create data profile: {str(e)}"
        else:
            state["error"] = "No valid DataFrame available for profiling"
        return state

    def classify_intent_node(self, state: AgentState) -> AgentState:
        """Classify user intent based on query and data profile (unverändert)"""
        query = state["query"]
        profile = state["data_profile"]
        
        if not profile:
            state["error"] = "Data profile is missing, cannot classify intent."
            return state

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are an expert query classifier.
            Classify intent: **analysis**, **visualization**, or **profile_qa**.
            - **analysis**: Calculating/analyzing data content (e.g., "What is the average age?").
            - **visualization**: User asks for a chart/plot.
            - **profile_qa**: User asks about the data's *structure* (e.g., "How many rows?").

            Data Profile Summary:
            - Shape: {profile.shape}
            - Columns: {profile.columns}
            Respond ONLY with the classification.
            """),
            HumanMessage(content=f"User Query: {query}")
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            intent = response.content.strip().lower()
            state["intent"] = intent if intent in ["analysis", "visualization", "profile_qa"] else "analysis"
        except Exception as e:
            state["error"] = f"Failed to classify query intent: {str(e)}"
        return state

    # ##################################################################
    # OPTIMIERUNG 4: Überarbeiteter `execute_analysis_node`
    # ##################################################################
    
    def execute_analysis_node(self, state: AgentState) -> AgentState:
        """Execute (simple or complex) analysis queries using a self-correcting ReAct loop."""
        
        df = self._get_dataframe(state)
        query = state["query"]
        profile = state["data_profile"]
        
        if df is None:
            state["error"] = "No DataFrame available for analysis"
            return state

        # Vereinfachter System-Prompt: LLM soll *nur* Code generieren.
        code_gen_system_prompt = SystemMessage(content="""You are a Python data analysis expert.
            Your task is to generate Python code to answer the user's question.
            
            IMPORTANT RULES:
            1.  The DataFrame is available as 'df'.
            2.  Store the final result in a variable called 'result'.
            3.  Your response MUST be *only* the raw Python code.
            4.  Do NOT include ```python```, import statements, or any explanations.
            
            Available DataFrame info:
            - Columns: {columns}
            - Shape: {shape}
            - Numeric columns: {numeric_columns}
            """.format(
                columns=profile.columns if profile else [],
                shape=profile.shape if profile else (0, 0),
                numeric_columns=profile.numeric_columns if profile else []
            ))

        # Chat-Verlauf für den ReAct-Loop
        messages: List[BaseMessage] = [
            code_gen_system_prompt,
            HumanMessage(content=query)
        ]
        
        df_json = df.to_json()

        for attempt in range(self.MAX_CODE_ATTEMPTS):
            try:
                # 1. GENERATE (LLM generiert reinen Code)
                print(f"--- Code Gen Attempt {attempt + 1} ---")
                ai_response = self.llm.invoke(messages)
                
                # Code extrahieren und bereinigen (falls LLM doch ```python``` hinzufügt)
                code_to_execute = self._extract_code(ai_response.content)

                if not code_to_execute:
                    raise ValueError("LLM returned empty code.")

                # 2. ACT (Python-Knoten führt Code-Tool aus)
                raw_result_str = execute_pandas_code.invoke({
                    "code": code_to_execute,
                    "df_json": df_json
                })
                
                # 3. ERFOLG!
                print("--- Code Execution Successful ---")
                
                synthesis_prompt = ChatPromptTemplate.from_messages([
                    SystemMessage("You are a helpful data analyst. Explain the result of the analysis in a clear, natural language answer."),
                    HumanMessage(f"Original Query: {query}"),
                    HumanMessage(f"Analysis Result (from code): {raw_result_str}"),
                    HumanMessage("Provide a final answer to the user based on the result.")
                ])
                final_answer = self.llm.invoke(synthesis_prompt.format_messages())
                
                state["analysis_result"] = AnalysisResult(
                    answer=final_answer.content,
                    confidence=0.9,
                    execution_time=0.5, # Placeholder
                    approach_used=f"code_execution (self-corrected {attempt} times)",
                    intermediate_results={"generated_code": code_to_execute, "raw_result": raw_result_str}
                )
                return state # Erfolgreich, den Knoten verlassen

            except Exception as e:
                # 4. REFLECT (Fehler behandeln)
                print(f"--- Code Execution Failed: {e} ---")
                
                # Fügen Sie die KI-Antwort (den fehlerhaften Code) und die Fehlermeldung zum Verlauf hinzu
                messages.append(AIMessage(content=code_to_execute)) # Der Code, der gescheitert ist
                messages.append(HumanMessage(
                    content=f"Your code produced an error: {str(e)}. Please fix the code and provide a new version. Remember: only output the raw code."
                ))
                # Die Schleife wird nun mit dem Fehler im Chat-Verlauf wiederholt
        
        # Wenn die Schleife ohne Erfolg endet
        state["error"] = f"Failed to execute code after {self.MAX_CODE_ATTEMPTS} attempts."
        return state

    # ##################################################################
    # Ende des optimierten Knotens
    # ##################################################################

    def execute_profile_qa_node(self, state: AgentState) -> AgentState:
        """Answer questions based *only* on the data profile (unverändert)"""
        query = state["query"]
        profile = state["data_profile"]
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage("You are a data assistant. Answer the user's question *using only* the provided data profile. Be concise."),
                HumanMessage(f"Data Profile:\n- Shape (rows, columns): {profile.shape}\n- Columns: {profile.columns}\n- Numeric columns: {profile.numeric_columns}\n\nQuery: {query}\nAnswer:")
            ])
            response = self.llm.invoke(prompt.format_messages())
            state["analysis_result"] = AnalysisResult(
                answer=response.content, confidence=0.95, execution_time=0.1, approach_used="profile_qa"
            )
        except Exception as e:
            state["error"] = str(e)
        return state

    def execute_visualization_node(self, state: AgentState) -> AgentState:
        """Generate visualizations based on the query (unverändert)"""
        df = self._get_dataframe(state)
        query = state["query"]
        if df is None:
            state["error"] = "No DataFrame available for visualization"
            return state
        try:
            viz_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content='Determine visualization. Return JSON: {"chart_type": "...", "config": {"x": "...", "y": "..."}}'),
                HumanMessage(content=f"Query: {query}\nColumns: {df.columns.tolist()}")
            ])
            viz_response = self.llm.invoke(viz_prompt.format_messages())
            viz_config = json.loads(viz_response.content)
            
            viz_result = generate_visualization.invoke({
                "df_json": df.to_json(),
                "chart_type": viz_config["chart_type"],
                "config": json.dumps(viz_config.get("config", {}))
            })
            
            state["analysis_result"] = AnalysisResult(
                answer=f"Generated {viz_config['chart_type']} visualization. (Chart JSON stored in state).",
                confidence=0.9, execution_time=0.3, approach_used="visualization",
                visualizations=[viz_result]
            )
        except Exception as e:
            state["error"] = str(e)
        return state
    
    def synthesize_result_node(self, state: AgentState) -> AgentState:
        """Synthesize the final result from analysis (unverändert)"""
        result = state.get("analysis_result")
        if result:
            state["messages"].append(AIMessage(content=result.answer))
            state["execution_history"].append({
                "timestamp": datetime.now().isoformat(), "query": state["query"],
                "approach": result.approach_used, "execution_time": result.execution_time
            })
        return state
    
    def handle_error_node(self, state: AgentState) -> AgentState:
        """Handle errors and potentially retry (unverändert)"""
        error = state.get("error")
        retry_count = state.get("retry_count", 0)
        state["retry_count"] = retry_count + 1
        state["messages"].append(AIMessage(content=f"Error occurred: {error}. Retry attempt {retry_count + 1}"))
        if retry_count < self.max_retries:
            state["error"] = None
        return state
    
    def route_by_intent(self, state: AgentState) -> str:
        """Route based on query intent (unverändert)"""
        if state.get("error"): return "error"
        intent = state.get("intent")
        if intent == "analysis": return "analysis"
        if intent == "visualization": return "visualization"
        if intent == "profile_qa": return "profile_qa"
        return "analysis" # Fallback
    
    def should_retry(self, state: AgentState) -> str:
        """Determine if we should retry after error (unverändert)"""
        if state.get("retry_count", 0) < self.max_retries:
            return "retry"
        return "end"
    
    def _extract_code(self, text: str) -> str:
        """Extract Python code from LLM response (jetzt wichtiger)"""
        if "```python" in text:
            code = text.split("```python")[1].split("```")[0]
        elif "```" in text:
            code = text.split("```")[1].split("```")[0]
        else:
            code = text
        return code.strip()
    

    def invoke(self, question: str, df: pd.DataFrame) -> str:
        """Main entry point for the agent (unverändert)"""
        self.current_dataframe = df
        initial_state: AgentState = {
            "messages": [], "dataframe_json": df.to_json(), "data_profile": None,
            "query": question, "intent": None, "analysis_result": None,
            "error": None, "retry_count": 0, "execution_history": [],
            "cache": {}, "metadata": {}
        }
        thread_id = hashlib.md5(f"{question}_{datetime.now()}".encode()).hexdigest()
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            final_state = self.app.invoke(initial_state, config)
            if final_state.get("analysis_result"):
                return final_state["analysis_result"].answer
            elif final_state.get("error"):
                return f"Error: {final_state['error']}"
            else:
                return "Unable to process the query"
        except Exception as e:
            return f"Critical error during graph execution: {str(e)}"

    
    def eval(self, question: str, dataset: pathlib.Path, additional_info: str) -> str:
        df = pd.read_csv(dataset, on_bad_lines='skip')
        full_question = f"{additional_info}\n{question}" if additional_info else question
        return self.invoke(full_question, df)
    
    async def ainvoke(self, question: str, df: pd.DataFrame) -> str:
        """Async version of invoke (unverändert)"""
        self.current_dataframe = df
        initial_state: AgentState = {
            "messages": [], "dataframe_json": df.to_json(), "data_profile": None,
            "query": question, "intent": None, "analysis_result": None,
            "error": None, "retry_count": 0, "execution_history": [],
            "cache": {}, "metadata": {}
        }
        thread_id = hashlib.md5(f"{question}_{datetime.now()}".encode()).hexdigest()
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            final_state = await self.app.ainvoke(initial_state, config)
            if final_state.get("analysis_result"):
                return final_state["analysis_result"].answer
            elif final_state.get("error"):
                return f"Error: {final_state['error']}"
            else:
                return "Unable to process the query"
        except Exception as e:
            return f"Critical error during graph execution: {str(e)}"

# ============= Streamlit UI (mit PyArrow-Fix) =============

def main():
    st.set_page_config(page_title="LangGraph Tabular Agent", layout="wide")
    st.title("🔄 Advanced LangGraph Data Analysis Agent")
    st.markdown("Upload a CSV file and ask complex questions using the power of LangGraph's workflow system.")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_name = st.selectbox("Model", ["mistral.mistral-small-2402-v1:0", "claude-3-sonnet", "gpt-4"], index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        max_retries = st.number_input("Max Retries", 1, 5, 3)
        max_tokens = st.number_input("Max Tokens", 100, 8000, 4096)
        
        st.divider()
        st.header("📊 Agent Features (Optimized)")
        st.markdown("""
        - **Intent Classification** (Analysis, Viz, QA)
        - **Self-Correcting Code Generation (ReAct)**
        - **Profile-based QA** (no code)
        - **Error Handling** with retry logic
        """)
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Rows", df.shape[0])
        with col2: st.metric("Columns", df.shape[1])
        with col3: st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col4: st.metric("Null Values", df.isnull().sum().sum())
        
        with st.expander("📋 Data Preview", expanded=True):
            tab1, tab2, tab3 = st.tabs(["First 10 Rows", "Data Info", "Statistics"])
            with tab1: st.dataframe(df.head(10))
            with tab2:
                info_df = pd.DataFrame({
                    'Column': df.columns, 
                    # OPTIMIERUNG 1: .astype(str) hinzugefügt, um PyArrow-Fehler zu beheben
                    'Type': df.dtypes.astype(str), 
                    'Non-Null Count': df.count(), 
                    'Null Count': df.isnull().sum(),
                    'Unique Values': df.nunique()
                })
                st.dataframe(info_df)
            with tab3: 
                try:
                    st.dataframe(df.describe())
                except Exception as e:
                    st.warning(f"Could not generate statistics (likely no numeric data): {e}")
        
        st.divider()
        
        with st.expander("💡 Example Questions"):
            st.markdown("""
            **Analysis (Code-Gen):**
            - What is the average value of column X?
            - How many unique values are in column Y?
            
            **Visualization (Code-Gen):**
            - Create a scatter plot of X vs Y
            
            **Profile QA (No-Code):**
            - How many rows and columns are in the data?
            """)
        
        user_question = st.text_area(
            "🤔 Ask a question about your data:",
            height=100,
            placeholder="E.g., What patterns exist in the data? Are there any significant correlations?"
        )
        
        if st.button("🚀 Analyze", type="primary", use_container_width=True) and user_question:
            with st.spinner("🧠 Thinking... (using Self-Correcting ReAct loop)"):
                try:
                    agent = LangGraphTabularAgent(
                        llm_model=model_name,
                        temperature=temperature,
                        max_retries=max_retries,
                        max_tokens=max_tokens
                    )
                    
                    result = agent.invoke(user_question, df)
                    
                    st.success("✅ Analysis Complete!")
                    st.markdown("### 📊 Result")
                    st.write(result)
                    
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    st.exception(e)
        
if __name__ == "__main__":
    main()