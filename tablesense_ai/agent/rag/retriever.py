from dotenv import load_dotenv
import os
import re
#import traceback
from typing import Optional, List, Dict
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

from smolagents import Tool, CodeAgent, LiteLLMModel
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# === Environment ===
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = "us-east-1"


# --- PostgreSQL Configuration ---
DB_USER = "tablesense_user"
DB_PASS = "tablesense_password"
DB_HOST = "localhost" # Connects to your host-mapped port
DB_PORT = "5433"
DB_NAME = "tablesense_db"
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# === Database Helper Class ===
class PostgreSQLHelper:
    """Manages SQLAlchemy Engine creation and schema fetching."""
    def __init__(self, db_url: str):
        self.db_url = db_url
        self._engine: Optional[Engine] = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(self.db_url)
        return self._engine

    def fetch_table_schema(self, table_name: str) -> str:
        """
        Fetch the schema (column names & types) for a table.
        Prints detailed error if inspection fails.
        """
        try:
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name)
            
            if not columns:
                # Check if the table exists but is empty (or permissions issue)
                if table_name not in inspector.get_table_names():
                     return f"TableNotFound: {table_name}"
                return f"TableSchemaEmpty: {table_name}"
            
            return ", ".join([f"{col['name']} ({col['type']})" for col in columns])
            
        except Exception as e:
            print("--- DETAILED POSTGRES SCHEMA FETCH ERROR ---")
            print(e)
            #traceback.print_exc()


# No write Queries are allowed!            
def is_select_query(sql: str) -> bool:
    sql_stripped = sql.strip().lower()
    return bool(re.match(r"^(with\s+.+\s+select|select)\b", sql_stripped))

# === SQL Query Tool ===
class SQLQueryTool(Tool):
    name = "sql_query"
    description = "Run SQL SELECT queries against any table in PostgreSQL."
    inputs = {
        "query": {"type": "string", "description": "SQL SELECT query."}
    }
    output_type = "string"

    def __init__(self, db_helper: PostgreSQLHelper, **kwargs):
        super().__init__(**kwargs)
        self.db_helper = db_helper

    def forward(self, query: str) -> str:
        if not isinstance(query, str):
            return "Input error: query must be a string."
        if not is_select_query(query):
            return "Rejected: only SELECT queries allowed."
        
        try:
            with self.db_helper.engine.connect() as conn:
                result = conn.execute(text(query))

                # Fetch all rows and column names for easy serialization
                col_names = list(result.keys())
                rows = result.fetchall()
                
                # Convert results to a list of dictionaries
                result_list = [dict(zip(col_names, row)) for row in rows]
                
            return str(result_list)
        except Exception as e:
            return f"SQL error: {e}"

# --- Retriever Tool with automatic schema ---
class RetrieverTool(Tool):
    name = "retriever"
    description = "Retrieve documentation and automatically append schema of any tables mentioned. Use it if you need more information on the user query"
    inputs = {"query": {"type": "string", "description": "Query for documents"}}
    output_type = "string"

    def __init__(self, db_helper: PostgreSQLHelper, **kwargs):
        super().__init__(**kwargs)
        self.db_helper = db_helper
        self.vectorstore = None

        try:
            self.vectorstore = Milvus(
                embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
                builtin_function=BM25BuiltInFunction(),
                vector_field=["dense", "sparse"],
                connection_args={"uri": "http://localhost:19530"},
                consistency_level="Bounded",
            )
        except Exception as e:
            self._init_error = f"Failed to initialize Milvus vectorstore: {e}"
            self.vectorstore = None

    def forward(self, query: str) -> str:
        if not query.strip():
            return "Input error: query must be non-empty."
        if self.vectorstore is None:
            return f"RetrieverUnavailable: {getattr(self, '_init_error', 'vectorstore not configured')}"
        try:
            docs = self.vectorstore.similarity_search(query, k=3, ranker_type="weighted", ranker_params={"weights": [0.6, 0.4]})
            if not docs:
                return "No relevant documents found."

            formatted_docs = []
            # List to hold unique schemas formatted for the LLM
            schema_output = [] 
            # Set to track table IDs whose schema has already been fetched
            processed_table_ids = set() 
            
            for i, doc in enumerate(docs, start=1):
                content = getattr(doc, "page_content", str(doc))
                meta = getattr(doc, "metadata", {})
                
                # Retrieve the full table name (e.g., 'table_baf49e0a3d')
                table_name = meta.get("table_id") if meta else None
                
                # Check if a table ID exists AND if we haven't processed it yet
                if table_name and table_name not in processed_table_ids:
                    
                    # 1. Fetch the schema using the helper
                    schema_str = self.db_helper.fetch_table_schema(table_name)
                    
                    # 2. Add the schema to the separate output list
                    schema_output.append(f"Table Name: {table_name}\nSchema: {schema_str}")
                    
                    # 3. Mark the table ID as processed
                    processed_table_ids.add(table_name)
                    
                # 4. Format and append the document content (WITHOUT the schema)
                formatted_docs.append(f"===== Document {i} =====\n{content}\nMetadata: {meta}")
            
            # 5. Join all document strings
            final_output = "\n".join(formatted_docs)
            
            # 6. Append the schemas at the very end if any were found
            if schema_output:
                final_output += "\n\n--- Schemas for Referenced Tables ---\n"
                final_output += "\n".join(schema_output)
                
            return final_output
        except Exception as e:
            return f"Error during retrieval: {e}"


# === Instantiate tools and agent ===
db_helper = PostgreSQLHelper(db_url=DB_URL)
sql_tool = SQLQueryTool(db_helper=db_helper)
retriever_tool = RetrieverTool(db_helper=db_helper)

llm_model = LiteLLMModel(
    model_id="bedrock/mistral.mistral-small-2402-v1:0",
    max_retries=2,
    max_tokens=4096,
)

agent = CodeAgent(
    tools=[retriever_tool, sql_tool],
    model=llm_model,
    max_steps=6,
    verbosity_level=2
)


# --- Agent Flow and Prompt ---

# "How many employees started in the Engineering department on or after January 1st, 2019?"
# "Do any of the top performers (A+) work remotely?"
# "How many projects Alice completed?"
# "What is the highest, lowest and average Salary (€)?"
question = "How many projects Alice completed?"

# Run retrieval manually to populate the input, as defined in your original script structure

retrieval_output = retriever_tool.forward(question)

    
prompt_base = """You are a data analyst assistant.

Workflow rules:
- Only if you need more information, retrieve additional documents.
- For simple look up question, you can diretly answer it if possible.
- As soon as it becomes more complex use sql_query for SELECT queries. This is the only code you are allowed to generate. The retrieved table snippets may not represent the whole table. So here you can refer to the table via the sql tool.

Reason step-by-step and concise.
"""

few_shot='''
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

agent_input = prompt_base + few_shot + "\n\nRetrieved:\n" + retrieval_output + "\n\nUser query: " + question
agent_output = agent.run(agent_input)

print("\n")
print(agent_output)

