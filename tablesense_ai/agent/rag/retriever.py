import os
import re
from typing import Optional, List, Dict
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

from smolagents import Tool
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_huggingface import HuggingFaceEmbeddings

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
                if table_name not in inspector.get_table_names():
                     return f"TableNotFound: {table_name}"
                return f"TableSchemaEmpty: {table_name}"
            
            return ", ".join([f"{col['name']} ({col['type']})" for col in columns])
            
        except Exception as e:
            print("--- DETAILED POSTGRES SCHEMA FETCH ERROR ---")
            print(e)
            return f"SchemaFetchError: {e}"

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
                col_names = list(result.keys())
                rows = result.fetchall()
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
        self._init_error = None # Initialize error tracking

        try:
            self.vectorstore = Milvus(
                embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
                builtin_function=BM25BuiltInFunction(),
                vector_field=["dense", "sparse"],
                connection_args={"host": "localhost", "port": "19530"},
                consistency_level="Bounded",
            )
        except Exception as e:
            self._init_error = f"Failed to initialize Milvus vectorstore: {e}"
            self.vectorstore = None

    def forward(self, query: str) -> str:
        if not query.strip():
            return "Input error: query must be non-empty."
        if self.vectorstore is None:
            return f"RetrieverUnavailable: {self._init_error or 'vectorstore not configured'}"
        
        try:
            docs = self.vectorstore.similarity_search(query, k=3, ranker_type="weighted", ranker_params={"weights": [0.6, 0.4]})
            if not docs:
                return "No relevant documents found."

            formatted_docs = []
            schema_output = [] 
            processed_table_ids = set() 
            
            for i, doc in enumerate(docs, start=1):
                content = getattr(doc, "page_content", str(doc))
                meta = getattr(doc, "metadata", {})
                table_name = meta.get("table_id") if meta else None
                
                if table_name and table_name not in processed_table_ids:
                    schema_str = self.db_helper.fetch_table_schema(table_name)
                    schema_output.append(f"Table Name: {table_name}\nSchema: {schema_str}")
                    processed_table_ids.add(table_name)
                    
                formatted_docs.append(f"===== Document {i} =====\n{content}\nMetadata: {meta}")
            
            final_output = "\n".join(formatted_docs)
            
            if schema_output:
                final_output += "\n\n--- Schemas for Referenced Tables ---\n"
                final_output += "\n".join(schema_output)
                
            return final_output
        except Exception as e:
            return f"Error during retrieval: {e}"
