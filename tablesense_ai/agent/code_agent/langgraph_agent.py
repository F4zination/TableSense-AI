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

# ============= Performance Monitoring =============

def langgraph_measure_performance(func):
    """Custom performance decorator for LangGraph agents"""
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(self, *args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log performance info
            print(f"LangGraph Agent executed in {execution_time:.2f} seconds")
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"LangGraph Agent failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper

# Mock monitor for compatibility with existing performance infrastructure
class MockMonitor:
    def get_total_token_counts(self):
        return {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

# ============= Data Models =============

class QueryComplexity(Enum):
    """Classification of query complexity"""
    SIMPLE = "simple"  # Basic lookups, counts, simple aggregations
    MODERATE = "moderate"  # Multiple operations, grouping, filtering
    COMPLEX = "complex"  # Statistical analysis, correlations, predictions
    VISUALIZATION = "visualization"  # Chart/plot generation

class DataProfile(BaseModel):
    """Data profiling results"""
    shape: tuple
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    memory_usage: float
    sample_data: Optional[List[Dict[str, Any]]] = None  # Fixed: Changed from Dict to List[Dict[str, Any]]

class QueryPlan(BaseModel):
    """Execution plan for the query"""
    complexity: QueryComplexity
    required_operations: List[str]
    suggested_approach: str
    needs_code_execution: bool
    estimated_steps: int

class AnalysisResult(BaseModel):
    """Result of data analysis"""
    answer: str
    confidence: float
    execution_time: float
    approach_used: str
    intermediate_results: Optional[Dict] = None
    visualizations: Optional[List[Dict]] = None

# ============= State Definition =============

class AgentState(TypedDict):
    """State schema for the LangGraph agent"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # dataframe: Optional[pd.DataFrame]  # Removed to avoid serialization issues
    dataframe_json: Optional[str]  # Store as JSON string instead
    data_profile: Optional[DataProfile]
    query: str
    query_plan: Optional[QueryPlan]
    analysis_result: Optional[AnalysisResult]
    error: Optional[str]
    retry_count: int
    execution_history: List[Dict[str, Any]]
    cache: Dict[str, Any]
    metadata: Dict[str, Any]

# ============= Tools Definition =============

@tool
def profile_dataframe(df_json: str) -> Dict:
    """Profile a pandas DataFrame to understand its structure and content"""
    df = pd.read_json(df_json)
    
    profile = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
        "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
        "sample_data": df.head(3).to_dict(orient='records')
    }
    
    # Add statistical summary for numeric columns
    if profile["numeric_columns"]:
        profile["numeric_summary"] = df[profile["numeric_columns"]].describe().to_dict()
    
    return profile

@tool
def execute_pandas_code(code: str, df_json: str) -> str:
    """Execute pandas code on a DataFrame and return the result"""
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import json
    import sys
    import io
    
    try:
        # Parse the DataFrame from JSON
        df = pd.read_json(df_json)
        
        # Create a more permissive execution environment
        # Include common modules that might be needed
        safe_globals = {
            'pd': pd,
            'pandas': pd,
            'np': np,
            'numpy': np,
            'datetime': datetime,
            'json': json,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'print': print,
            '__builtins__': {
                '__import__': __import__,  # Allow imports
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'print': print,
            }
        }
        
        # Add the DataFrame to the execution environment
        safe_globals['df'] = df
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # Execute the code
            exec(code, safe_globals)
            
            # Get captured print output
            printed_output = captured_output.getvalue()
            
            # Try to get the result variable
            if 'result' in safe_globals:
                result = safe_globals['result']
                
                # Format the result based on its type
                if isinstance(result, pd.DataFrame):
                    if len(result) > 100:  # Limit large results
                        result_str = f"DataFrame with {len(result)} rows and {len(result.columns)} columns:\n"
                        result_str += result.head(10).to_string()
                        result_str += f"\n... (showing first 10 of {len(result)} rows)"
                    else:
                        result_str = result.to_string()
                elif isinstance(result, pd.Series):
                    if len(result) > 50:  # Limit large series
                        result_str = f"Series with {len(result)} values:\n"
                        result_str += result.head(10).to_string()
                        result_str += f"\n... (showing first 10 of {len(result)} values)"
                    else:
                        result_str = result.to_string()
                elif hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
                    # Handle lists, tuples, etc.
                    if len(result) > 20:
                        result_str = f"Collection with {len(result)} items: {list(result)[:10]}... (truncated)"
                    else:
                        result_str = str(result)
                else:
                    result_str = str(result)
                
                # Combine printed output with result
                if printed_output:
                    return f"{printed_output}\nResult: {result_str}"
                else:
                    return result_str
            else:
                # No result variable, return printed output or success message
                if printed_output:
                    return printed_output
                else:
                    return "Code executed successfully (no 'result' variable defined and no output printed)"
                    
        finally:
            # Restore stdout
            sys.stdout = old_stdout
            
    except SyntaxError as e:
        return f"Syntax Error: {str(e)}"
    except NameError as e:
        return f"Name Error: {str(e)}. Make sure all variables and functions are properly defined."
    except KeyError as e:
        return f"Key Error: {str(e)}. Check that column names exist in the DataFrame."
    except Exception as e:
        return f"Error executing code: {str(e)}"

@tool
def generate_visualization(df_json: str, chart_type: str, config: str) -> Dict:
    """Generate a visualization from the DataFrame"""
    import plotly.express as px
    import plotly.graph_objects as go
    import json
    
    df = pd.read_json(df_json)
    config_dict = json.loads(config)
    
    try:
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

@tool
def statistical_analysis(df_json: str, analysis_type: str, columns: List[str]) -> Dict:
    """Perform statistical analysis on DataFrame columns"""
    import scipy.stats as stats
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    df = pd.read_json(df_json)
    
    try:
        if analysis_type == "correlation":
            if len(columns) >= 2:
                result = df[columns].corr().to_dict()
            else:
                result = df.corr().to_dict()
        
        elif analysis_type == "distribution":
            result = {}
            for col in columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    result[col] = {
                        "mean": df[col].mean(),
                        "std": df[col].std(),
                        "skew": df[col].skew(),
                        "kurtosis": df[col].kurtosis(),
                        "quartiles": df[col].quantile([0.25, 0.5, 0.75]).to_dict()
                    }
        
        elif analysis_type == "outliers":
            result = {}
            for col in columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                    result[col] = {
                        "count": len(outliers),
                        "percentage": len(outliers) / len(df) * 100,
                        "values": outliers[col].tolist()[:10]  # First 10 outliers
                    }
        
        elif analysis_type == "pca":
            numeric_cols = df[columns].select_dtypes(include=[np.number])
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_cols.dropna())
            pca = PCA(n_components=min(3, len(numeric_cols.columns)))
            pca_result = pca.fit_transform(scaled_data)
            result = {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "components": pca.components_.tolist(),
                "n_components": pca.n_components_
            }
        
        else:
            result = {"error": f"Unknown analysis type: {analysis_type}"}
        
        return result
    
    except Exception as e:
        return {"error": str(e)}

# ============= LangGraph Agent Implementation =============

class LangGraphTabularAgent(BaseAgent):
    def __init__(self, llm_model, temperature=0, max_retries=3, max_tokens=4096):
        super().__init__(llm_model, temperature, max_retries, max_tokens, base_url, api_key)
        
        # Add mock code_agent with monitor for compatibility with performance monitoring
        self.code_agent = type('MockCodeAgent', (), {'monitor': MockMonitor()})()
        
        # Initialize LLM
        self.llm = ChatLiteLLM(
            model=model_id,
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens,
            api_base=base_url,
            api_key=api_key
        )
        
        # Create tools
        self.tools = [
            profile_dataframe,
            execute_pandas_code,
            generate_visualization,
            statistical_analysis
        ]
        
        # Initialize the graph
        self.graph = self._build_graph()
        
        # Store DataFrame separately (not in checkpointed state)
        self.current_dataframe = None
        
        # Compile without checkpointer to avoid serialization issues
        self.app = self.graph.compile()
        
    def _get_dataframe(self, state: AgentState) -> pd.DataFrame:
        """Safely retrieve DataFrame from instance or state"""
        # First try to get from instance variable
        if hasattr(self, 'current_dataframe') and self.current_dataframe is not None:
            return self.current_dataframe
        
        # Fallback to state if available
        if state.get("dataframe_json"):
            try:
                return pd.read_json(state["dataframe_json"])
            except Exception:
                pass
        
        return None
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self.analyze_query_node)
        workflow.add_node("profile_data", self.profile_data_node)
        workflow.add_node("plan_execution", self.plan_execution_node)
        workflow.add_node("execute_simple", self.execute_simple_node)
        workflow.add_node("execute_complex", self.execute_complex_node)
        workflow.add_node("execute_visualization", self.execute_visualization_node)
        workflow.add_node("synthesize_result", self.synthesize_result_node)
        workflow.add_node("handle_error", self.handle_error_node)
        
        # Add conditional edges
        workflow.add_edge(START, "analyze_query")
        workflow.add_edge("analyze_query", "profile_data")
        workflow.add_edge("profile_data", "plan_execution")
        
        # Conditional routing based on query complexity
        workflow.add_conditional_edges(
            "plan_execution",
            self.route_by_complexity,
            {
                "simple": "execute_simple",
                "complex": "execute_complex",
                "visualization": "execute_visualization",
                "error": "handle_error"
            }
        )
        
        # All execution nodes lead to synthesis
        workflow.add_edge("execute_simple", "synthesize_result")
        workflow.add_edge("execute_complex", "synthesize_result")
        workflow.add_edge("execute_visualization", "synthesize_result")
        
        # Error handling can retry or end
        workflow.add_conditional_edges(
            "handle_error",
            self.should_retry,
            {
                "retry": "plan_execution",
                "end": END
            }
        )
        
        # Synthesis leads to end
        workflow.add_edge("synthesize_result", END)
        
        return workflow
    
    def analyze_query_node(self, state: AgentState) -> AgentState:
        """Analyze the user query to understand intent"""
        query = state["query"]
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a query analyzer for tabular data operations.
            Analyze the user's query and identify:
            1. The main intent (lookup, aggregation, calculation, visualization, etc.)
            2. Key entities mentioned (column names, values, operations)
            3. Expected output format"""),
            HumanMessage(content=f"Query: {query}")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        state["messages"].append(HumanMessage(content=query))
        state["messages"].append(response)
        state["metadata"]["query_analysis"] = response.content
        
        return state
    
    def profile_data_node(self, state: AgentState) -> AgentState:
        """Profile the DataFrame to understand its structure"""
        
        df = self._get_dataframe(state)
        
        if df is not None and isinstance(df, pd.DataFrame):
            try:
                profile = DataProfile(
                    shape=df.shape,
                    columns=df.columns.tolist(),
                    dtypes=df.dtypes.astype(str).to_dict(),
                    missing_values=df.isnull().sum().to_dict(),
                    numeric_columns=df.select_dtypes(include=[np.number]).columns.tolist(),
                    categorical_columns=df.select_dtypes(include=['object', 'category']).columns.tolist(),
                    datetime_columns=df.select_dtypes(include=['datetime64']).columns.tolist(),
                    memory_usage=df.memory_usage(deep=True).sum() / 1024**2,
                    sample_data=df.head(3).to_dict(orient='records')
                )
                
                state["data_profile"] = profile
                state["metadata"]["profiling_complete"] = True
            except Exception as e:
                state["error"] = f"Failed to create data profile: {str(e)}"
        else:
            state["error"] = "No valid DataFrame available for profiling"
        
        return state
    
    def plan_execution_node(self, state: AgentState) -> AgentState:
        """Plan the execution strategy based on query complexity"""
        query = state["query"]
        profile = state["data_profile"]
        
        # Use LLM to classify query complexity and plan execution
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a query planner for data analysis.
            Based on the query and data profile, determine:
            1. Query complexity: simple, moderate, complex, or visualization
            2. Required operations (filtering, grouping, aggregation, etc.)
            3. Whether code execution is needed
            4. Suggested approach
            
            Return a JSON object with these fields:
            {
                "complexity": "simple|moderate|complex|visualization",
                "required_operations": ["operation1", "operation2"],
                "needs_code_execution": true|false,
                "suggested_approach": "description",
                "estimated_steps": number
            }"""),
            HumanMessage(content=f"""
            Query: {query}
            
            Data Profile:
            - Shape: {profile.shape if profile else 'Unknown'}
            - Columns: {profile.columns[:10] if profile else 'Unknown'}
            - Numeric columns: {profile.numeric_columns[:10] if profile else 'Unknown'}
            """)
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        try:
            plan_dict = json.loads(response.content)
            query_plan = QueryPlan(
                complexity=QueryComplexity(plan_dict.get("complexity", "moderate")),
                required_operations=plan_dict.get("required_operations", []),
                suggested_approach=plan_dict.get("suggested_approach", ""),
                needs_code_execution=plan_dict.get("needs_code_execution", True),
                estimated_steps=plan_dict.get("estimated_steps", 1)
            )
            state["query_plan"] = query_plan
        except Exception as e:
            state["error"] = f"Failed to create execution plan: {str(e)}"
        
        return state
    
    def execute_simple_node(self, state: AgentState) -> AgentState:
        """Execute simple queries using direct DataFrame operations"""
        df = self._get_dataframe(state)
        query = state["query"]
        
        if df is None:
            state["error"] = "No DataFrame available for analysis"
            return state
        
        try:
            # For simple queries, use a template-based approach
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a data analyst. 
                Answer the question directly based on the data summary provided.
                Be concise and specific."""),
                HumanMessage(content=f"""
                Query: {query}
                
                Data Summary:
                {df.describe().to_string() if len(df) < 1000 else df.head(100).describe().to_string()}
                
                Sample rows:
                {df.head(10).to_string()}
                """)
            ])
            
            response = self.llm.invoke(prompt.format_messages())
            
            state["analysis_result"] = AnalysisResult(
                answer=response.content,
                confidence=0.9,
                execution_time=0.1,
                approach_used="simple_analysis"
            )
        except Exception as e:
            state["error"] = str(e)
        
        return state
    
    def execute_complex_node(self, state: AgentState) -> AgentState:
        """Execute complex queries using code generation"""
        df = self._get_dataframe(state)
        query = state["query"]
        profile = state["data_profile"]
        
        if df is None:
            state["error"] = "No DataFrame available for analysis"
            return state
        
        try:
            # Generate pandas code with better instructions
            code_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a Python data analysis expert.
                Generate pandas code to answer the user's question.
                
                IMPORTANT RULES:
                1. Store the final result in a variable called 'result'
                2. Use only pandas (pd), numpy (np), and standard library functions
                3. The DataFrame is already available as 'df'
                4. Do NOT include any import statements
                5. Make sure to handle potential errors (missing columns, empty data, etc.)
                6. For statistical operations, ensure the data is numeric
                7. Use descriptive variable names
                
                Available DataFrame info:
                - Columns: {columns}
                - Shape: {shape}
                - Numeric columns: {numeric_columns}
                - Categorical columns: {categorical_columns}
                
                Example format:
                ```python
                # Your analysis code here
                result = df.mean()  # Example
                ```
                """.format(
                    columns=profile.columns if profile else [],
                    shape=profile.shape if profile else (0, 0),
                    numeric_columns=profile.numeric_columns if profile else [],
                    categorical_columns=profile.categorical_columns if profile else []
                )),
                HumanMessage(content=f"Question: {query}\n\nGenerate Python code (no imports needed):")
            ])
            
            code_response = self.llm.invoke(code_prompt.format_messages())
            
            # Extract and clean code from response
            code = self._extract_code(code_response.content)
            
            # Remove any import statements from the generated code
            code_lines = code.split('\n')
            cleaned_lines = []
            for line in code_lines:
                stripped_line = line.strip()
                if not (stripped_line.startswith('import ') or 
                       stripped_line.startswith('from ') or
                       stripped_line == ''):
                    cleaned_lines.append(line)
                elif stripped_line == '':
                    cleaned_lines.append(line)  # Keep empty lines for formatting
            
            cleaned_code = '\n'.join(cleaned_lines)
            
            # Execute the cleaned code
            result = execute_pandas_code.invoke({
                "code": cleaned_code,
                "df_json": df.to_json()
            })
            
            state["analysis_result"] = AnalysisResult(
                answer=str(result),
                confidence=0.85,
                execution_time=0.5,
                approach_used="code_execution",
                intermediate_results={"generated_code": cleaned_code, "original_code": code}
            )
            
        except Exception as e:
            state["error"] = f"Code execution failed: {str(e)}"
        
        return state
    
    def execute_visualization_node(self, state: AgentState) -> AgentState:
        """Generate visualizations based on the query"""
        df = self._get_dataframe(state)
        query = state["query"]
        
        if df is None:
            state["error"] = "No DataFrame available for visualization"
            return state
        
        try:
            # Determine visualization type and config
            viz_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""Determine the best visualization for the user's request.
                Return a JSON object with:
                {
                    "chart_type": "scatter|bar|line|histogram|box|heatmap",
                    "config": {
                        "x": "column_name",
                        "y": "column_name",
                        "color": "optional_column",
                        "title": "chart_title"
                    }
                }"""),
                HumanMessage(content=f"Query: {query}\nColumns: {df.columns.tolist()}")
            ])
            
            viz_response = self.llm.invoke(viz_prompt.format_messages())
            viz_config = json.loads(viz_response.content)
            
            # Generate visualization
            viz_result = generate_visualization.invoke({
                "df_json": df.to_json(),
                "chart_type": viz_config["chart_type"],
                "config": json.dumps(viz_config["config"])
            })
            
            state["analysis_result"] = AnalysisResult(
                answer=f"Generated {viz_config['chart_type']} visualization",
                confidence=0.9,
                execution_time=0.3,
                approach_used="visualization",
                visualizations=[viz_result]
            )
            
        except Exception as e:
            state["error"] = str(e)
        
        return state
    
    def synthesize_result_node(self, state: AgentState) -> AgentState:
        """Synthesize the final result from analysis"""
        result = state.get("analysis_result")
        
        if result:
            # Add any post-processing or formatting here
            state["messages"].append(
                AIMessage(content=result.answer)
            )
            
            # Log execution history
            state["execution_history"].append({
                "timestamp": datetime.now().isoformat(),
                "query": state["query"],
                "approach": result.approach_used,
                "execution_time": result.execution_time,
                "confidence": result.confidence
            })
        
        return state
    
    def handle_error_node(self, state: AgentState) -> AgentState:
        """Handle errors and potentially retry"""
        error = state.get("error")
        retry_count = state.get("retry_count", 0)
        
        state["retry_count"] = retry_count + 1
        state["messages"].append(
            AIMessage(content=f"Error occurred: {error}. Retry attempt {retry_count + 1}")
        )
        
        # Clear error for retry
        if retry_count < self.max_retries:
            state["error"] = None
        
        return state
    
    def route_by_complexity(self, state: AgentState) -> str:
        """Route based on query complexity"""
        plan = state.get("query_plan")
        error = state.get("error")
        
        if error:
            return "error"
        
        if not plan:
            return "simple"
        
        if plan.complexity == QueryComplexity.SIMPLE:
            return "simple"
        elif plan.complexity == QueryComplexity.VISUALIZATION:
            return "visualization"
        else:
            return "complex"
    
    def should_retry(self, state: AgentState) -> str:
        """Determine if we should retry after error"""
        retry_count = state.get("retry_count", 0)
        
        if retry_count < self.max_retries:
            return "retry"
        return "end"
    
    def _extract_code(self, text: str) -> str:
        """Extract Python code from LLM response"""
        # Look for code blocks
        if "```python" in text:
            code = text.split("```python")[1].split("```")[0]
        elif "```" in text:
            code = text.split("```")[1].split("```")[0]
        else:
            code = text
        
        return code.strip()
    

    def invoke(self, question: str, df: pd.DataFrame) -> str:
        """Main entry point for the agent"""
        
        # Store DataFrame in instance variable instead of state
        self.current_dataframe = df
        
        # Initialize state without DataFrame
        initial_state: AgentState = {
            "messages": [],
            "dataframe_json": None,  # Or optionally: df.to_json() if you want to store it
            "data_profile": None,
            "query": question,
            "query_plan": None,
            "analysis_result": None,
            "error": None,
            "retry_count": 0,
            "execution_history": [],
            "cache": {},
            "metadata": {}
        }
        
        # Create a thread ID for conversation tracking
        thread_id = hashlib.md5(f"{question}_{datetime.now()}".encode()).hexdigest()
        
        # Run the graph
        config = {"configurable": {"thread_id": thread_id}}
        final_state = self.app.invoke(initial_state, config)
        
        # Extract result
        if final_state.get("analysis_result"):
            return final_state["analysis_result"].answer
        elif final_state.get("error"):
            return f"Error: {final_state['error']}"
        else:
            return "Unable to process the query"
    
    def eval(self, question: str, dataset: pathlib.Path, additional_info: str) -> str:
        """Evaluation method compatible with base agent"""
        df = pd.read_csv(dataset, on_bad_lines='skip')
        full_question = f"{additional_info}\n{question}" if additional_info else question
        return self.invoke(full_question, df)
    
    async def ainvoke(self, question: str, df: pd.DataFrame) -> str:
        """Async version of invoke"""
        # Store DataFrame in instance variable
        self.current_dataframe = df
        
        # Initialize state
        initial_state: AgentState = {
            "messages": [],
            "dataframe_json": None,
            "data_profile": None,
            "query": question,
            "query_plan": None,
            "analysis_result": None,
            "error": None,
            "retry_count": 0,
            "execution_history": [],
            "cache": {},
            "metadata": {}
        }
        
        # Create a thread ID
        thread_id = hashlib.md5(f"{question}_{datetime.now()}".encode()).hexdigest()
        
        # Run the graph asynchronously
        config = {"configurable": {"thread_id": thread_id}}
        final_state = await self.app.ainvoke(initial_state, config)
        
        # Extract result
        if final_state.get("analysis_result"):
            return final_state["analysis_result"].answer
        elif final_state.get("error"):
            return f"Error: {final_state['error']}"
        else:
            return "Unable to process the query"

# ============= Streamlit UI =============

def main():
    st.set_page_config(page_title="LangGraph Tabular Agent", layout="wide")
    st.title("🔄 Advanced LangGraph Data Analysis Agent")
    st.markdown("Upload a CSV file and ask complex questions using the power of LangGraph's workflow system.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_name = st.selectbox(
            "Model",
            ["mistral.mistral-small-2402-v1:0", "claude-3-sonnet", "gpt-4"],
            index=0
        )
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        max_retries = st.number_input("Max Retries", 1, 5, 3)
        max_tokens = st.number_input("Max Tokens", 100, 8000, 4096)
        
        st.divider()
        st.header("📊 Agent Features")
        st.markdown("""
        - **Multi-step reasoning** with state management
        - **Automatic query routing** based on complexity
        - **Code generation** for complex analysis
        - **Statistical analysis** tools
        - **Visualization generation**
        - **Error handling** with retry logic
        - **Conversation memory**
        """)
    
    # Main content
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        # Load and display data
        df = pd.read_csv(uploaded_file)
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col4:
            st.metric("Null Values", df.isnull().sum().sum())
        
        # Data preview
        with st.expander("📋 Data Preview", expanded=True):
            tab1, tab2, tab3 = st.tabs(["First 10 Rows", "Data Info", "Statistics"])
            
            with tab1:
                st.dataframe(df.head(10))
            
            with tab2:
                info_df = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Unique Values': df.nunique()
                })
                st.dataframe(info_df)
            
            with tab3:
                st.dataframe(df.describe())
        
        # Query interface
        st.divider()
        
        # Query examples
        with st.expander("💡 Example Questions"):
            st.markdown("""
            **Simple queries:**
            - What is the average value of column X?
            - How many unique values are in column Y?
            - Show me the top 5 rows sorted by column Z
            
            **Complex analysis:**
            - What is the correlation between columns A and B?
            - Identify outliers in the numeric columns
            - Perform a statistical test comparing two groups
            
            **Visualizations:**
            - Create a scatter plot of X vs Y
            - Show me a distribution histogram of column Z
            - Generate a correlation heatmap
            """)
        
        # Query input
        user_question = st.text_area(
            "🤔 Ask a question about your data:",
            height=100,
            placeholder="E.g., What patterns exist in the data? Are there any significant correlations?"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)
        
        # Process query
        if analyze_button and user_question:
            with st.spinner("🧠 Thinking... (using LangGraph workflow)"):
                try:
                    # Initialize agent
                    agent = LangGraphTabularAgent(
                        llm_model=model_name,
                        temperature=temperature,
                        max_retries=max_retries,
                        max_tokens=max_tokens
                    )
                    
                    # Get result
                    result = agent.invoke(user_question, df)
                    
                    # Display result
                    st.success("✅ Analysis Complete!")
                    st.markdown("### 📊 Result")
                    st.write(result)
                    
                    # Show execution details in expander
                    with st.expander("🔍 Execution Details"):
                        if hasattr(agent, 'app'):
                            st.json({
                                "model": model_name,
                                "temperature": temperature,
                                "approach": "LangGraph Multi-Step Workflow",
                                "features_used": [
                                    "Query Analysis",
                                    "Data Profiling", 
                                    "Execution Planning",
                                    "Dynamic Routing"
                                ]
                            })
                
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    st.exception(e)
        
        # Additional features section
        with st.expander("🛠️ Advanced Features", expanded=False):
            st.markdown("""
            ### Batch Processing
            Upload multiple questions to process in batch:
            """)
            
            batch_questions = st.text_area(
                "Enter questions (one per line):",
                height=150,
                placeholder="What is the mean of column A?\nShow correlation matrix\nIdentify outliers"
            )
            
            if st.button("🔄 Run Batch Analysis"):
                if batch_questions:
                    questions = [q.strip() for q in batch_questions.split('\n') if q.strip()]
                    
                    with st.spinner(f"Processing {len(questions)} questions..."):
                        agent = LangGraphTabularAgent(
                            llm_model=model_name,
                            temperature=temperature,
                            max_retries=max_retries,
                            max_tokens=max_tokens
                        )
                        
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, question in enumerate(questions):
                            try:
                                answer = agent.invoke(question, df)
                                results.append({"Question": question, "Answer": answer})
                            except Exception as e:
                                results.append({"Question": question, "Answer": f"Error: {str(e)}"})
                            
                            progress_bar.progress((i + 1) / len(questions))
                        
                        # Display batch results
                        st.markdown("### Batch Results")
                        for i, result in enumerate(results, 1):
                            with st.container():
                                st.markdown(f"**Q{i}:** {result['Question']}")
                                st.markdown(f"**A:** {result['Answer']}")
                                st.divider()

if __name__ == "__main__":
    main()