from smolagents import Tool
from dotenv import load_dotenv
import os
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_huggingface import HuggingFaceEmbeddings
from smolagents import CodeAgent, LiteLLMModel
import sqlite3

load_dotenv()
#api_base = os.getenv("API_BASE")
#api_key = os.getenv("API_KEY")
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = "us-east-1"


class SQLQueryTool(Tool):
    name = "sql_query"
    description = "Run SQL queries against structured tables stored in SQLite. Use when you need exact numeric or categorical answers."
    inputs = {
        "query": {
            "type": "string",
            "description": "A valid SQL SELECT query. Example: SELECT Salary FROM table_baf49e0a3d WHERE Employee = 'Alice'"
        }
    }
    output_type = "string"

    def __init__(self, db_path: str = "tables.db", **kwargs):
        super().__init__(**kwargs)
        self.db_path = db_path
        #schema = self.get_table_schema(table_id)
        #print(schema)

    def forward(self, query: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]
            result = [dict(zip(col_names, row)) for row in rows]
        except Exception as e:
            result = f"SQL error: {e}"
        conn.close()
        return str(result)


    def get_table_schema(self, table_id: str):
        """
        TODO: 
        """
        table_name = f"table_{table_id}"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        cols = cursor.fetchall()  # (cid, name, type, notnull, dflt_value, pk)
        conn.close()
        # Strip trailing/leading spaces from column names
        return [(col[1].strip(), col[2]) for col in cols]


class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses hybrid search to retrieve the parts of documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically and lexically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vectorstore = Milvus(
            embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
            builtin_function=BM25BuiltInFunction(),  # output_field_names="sparse"),
            vector_field=["dense", "sparse"],
            connection_args={
                "uri": "http://localhost:19530",
            },
            consistency_level="Bounded",  #`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`. See https://milvus.io/docs/consistency.md#Consistency-Level for more details.
            # index params --> define metrics, e.g. cosine, ..
        )

    def forward(self, query: str):
        """Execute the retrieval based on the provided query."""
        assert isinstance(query, str), "search query must be a string"

        # Retrieve relevant documents --> evaluate reranker types
        docs = self.vectorstore.similarity_search(
            query, k=2, ranker_type="weighted", ranker_params={"weights": [0.6, 0.4]}
        )
        # Check how to format and present (e.g. with metadata, ...)
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {i} =====\n{doc.page_content}\nMetadata: {doc.metadata}"
                for i, doc in enumerate(docs)
            ]
        )

retriever_tool = RetrieverTool()
sql_tool = SQLQueryTool()

llm_old = LiteLLMModel(
        model_id="bedrock/mistral.mistral-small-2402-v1:0",
        max_retries=1,
        max_tokens=4096,
        )

agent = CodeAgent(
    tools=[sql_tool], 
    model=llm_old,  
    max_steps=4,  # number of reasoning steps
    verbosity_level=2,
    additional_authorized_imports=["duckdb"]
)


question = "What is the average number of completed projects?"

retrieved_result = retriever_tool(question)


# Optionally append data schema here
prompt = """
You are a data analyst.
Your job is to answer questions asked by the user. Those questions can be related to tables.

If the answer does not directly follow from the retriever tool with it's serialized table content,
you may use the SQL tool or ask questions to the ueser is something is unclear.
Don't combine the tools, make it step by step and decide if the sql tool is realy needed. You don't need to generate code for everything!

Here is more information from the retriever tool:
"""


agent_output = agent.run(prompt + retrieved_result + "\n\n" + "If tis snippet is not enough, use the sql tool and table_id to refer to the SQL table" +"\n\n" + "User query: " + question)

print("\nFinal answer:")
print(agent_output)


# direct SQL test
#sql_test_query = 'SELECT AVG("Projects Completed") as avg_projects FROM table_baf49e0a3d;'
#print(sql_tool.forward(sql_test_query))



## Smolagent Multiagent:
## https://huggingface.co/docs/smolagents/examples/multiagents
# retriever tool
# SQL Agent (capable of correcting code)
#  - Make SQL Agent instread of tool
# judge final answer (mayber steps for rethinking)

