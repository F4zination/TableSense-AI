import hashlib
import re
import pandas as pd
from io import StringIO
from langchain.docstore.document import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_huggingface import HuggingFaceEmbeddings
from tablesense_ai.agent.serialization.converter import Converter, TableFormat
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import sqlite3
from sqlalchemy import create_engine

POSTGRES_URL = "postgresql+psycopg2://tablesense_user:tablesense_password@localhost:5433/tablesense_db"

def store_table_in_sql(df: pd.DataFrame, table_id: str, db_url: str = POSTGRES_URL):
    """
    Store a DataFrame as a SQL table with inferred dtypes for exact querying
    using SQLAlchemy and PostgreSQL.
    Table name = table_<table_id>.
    """
    # 1. Create the SQLAlchemy Engine
    # Note: If running this function from *inside* a Docker container, 
    # replace 'localhost:5433' with 'postgres_db:5432'
    try:
        engine = create_engine(db_url)
    except Exception as e:
        print(f"Failed to create SQLAlchemy engine with URL: {db_url}. Error: {e}")
        return # Stop execution if connection setup fails

    table_name = f"table_{table_id}"

    # Clean column names: remove leading/trailing whitespace
    df_clean = df.copy()
    df_clean.columns = [c.strip() for c in df_clean.columns]

    # Try to clean numeric columns
    for col in df_clean.columns:
        # Remove commas and percent signs, then try numeric conversion
        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
        )
        # Convert to numeric if possible
        df_clean[col] = pd.to_numeric(df_clean[col], errors="ignore")

    # 2. Use the Engine to write the DataFrame to PostgreSQL
    try:
        # The 'with engine.connect()' handles opening and closing the connection
        with engine.connect() as conn:
            # pandas.to_sql handles the data type mapping and table creation/replacement
            df_clean.to_sql(table_name, conn, if_exists="replace", index=False)
        
        print(f"Stored table in PostgreSQL as '{table_name}' ({len(df_clean)} rows)")
    except Exception as e:
        # This catches errors during the connection or the actual table writing process
        print(f"Failed to store table {table_id} in PostgreSQL: {e}")
    # No 'finally' block needed, as 'with engine.connect()' closes the connection automatically.



def extract_markdown_tables(markdown_text: str):
    pattern = re.compile(r"(?:\|.+\|\n)+", re.MULTILINE)
    return [(match.group(), match.start(), match.end()) for match in pattern.finditer(markdown_text)]


def markdown_table_to_df(table_markdown: str) -> pd.DataFrame:
    try:
        return (
            pd.read_csv(
                StringIO(table_markdown),
                sep="|",
                engine="python",
            )
            .dropna(axis=1, how="all")  # drop empty cols
            .iloc[1:]  # skip the separator row (---)
        )
    except Exception as e:
        print(f"Failed to parse table: {e}")
        return None

def df_to_natural(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame into natural language sentences using Jinja2.
    """
    try:
        template_dir = Path(__file__).resolve().parent / "jinja_templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template('only_header.natural.jinja2')
        return template.render(dataframe=df)
    except Exception as e:
        print(f"Template rendering failed ({e}), using fallback.")

        # Fallback: describe first rows in simple sentences
        sentences = []
        for i, row in df.head(5).iterrows():
            parts = [f"{col} is '{row[col]}'" for col in df.columns]
            sentences.append(", ".join(parts))
        return "\n".join(sentences)

def chunk_table(df: pd.DataFrame, table_id: str, rows_per_chunk: int = 5, natural: bool = False):
    """
    Split DataFrame into chunks of N rows, return list of Documents.
    If natural=True, also produce natural language sentences.
    """
    documents = []
    for start in range(0, len(df), rows_per_chunk):
        chunk = df.iloc[start:start+rows_per_chunk]

        if natural:
            chunk_text = df_to_natural(chunk)
        else:
            chunk_text = chunk.to_markdown(index=False)

        metadata = {
            "table_id": f"table_{table_id}",
            "row_range": f"{start}-{start+len(chunk)-1}",
            "mode": "natural language don't execute code to parse it!!" if natural else "table"
        }
        documents.append(Document(page_content=chunk_text, metadata=metadata))
    return documents


def process_markdown_doc(doc: Document, rows_per_chunk: int = 5, natural: bool = False):
    """
    Process a LangChain Document containing markdown text,
    extract all tables, chunk them, and return new Documents.
    natural=True => generates sentences instead of markdown.
    """
    docs_out = []
    tables = extract_markdown_tables(doc.page_content)

    for table_markdown, start, end in tables:
        table_id = hashlib.md5(table_markdown.encode()).hexdigest()[:10]
        df = markdown_table_to_df(table_markdown)
        if df is not None:
            store_table_in_sql(df, table_id)
            docs_out.extend(chunk_table(df, table_id, rows_per_chunk, natural=natural))

    return docs_out


def index_documents(docs: list[Document]):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    Milvus.from_documents(
        documents=docs,
        embedding=embedding,
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        connection_args={"uri": "http://localhost:19530"},
        consistency_level="Bounded",
        drop_old=True,
    )


# === Example usage ===
if __name__ == "__main__":
    sample_markdown= """
# Example Markdown

## Employee Performance Overview

| Employee | Department   | Start Date | Salary (€) | Bonus % | Remote | Projects Completed | Performance Rating |
|----------|--------------|------------|------------|---------|--------|--------------------|--------------------|
| Alice    | Engineering  | 2020-01-15 | 65,000     | 10%     | True   | 12                 | A                  |
| Bob      | Engineering  | 2019-03-22 | 72,000     | 12%     | False  | 15                 | A+                 |
| Carol    | HR           | 2021-07-01 | 50,000     | 5%      | True   | 7                  | B                  |
| David    | Marketing    | 2018-11-10 | 58,500     | 7%      | False  | 9                  | B+                 |
| Eve      | Finance      | 2020-09-05 | 80,000     | 15%     | True   | 20                 | A+                 |
| Frank    | Engineering  | 2017-04-17 | 95,000     | 20%     | False  | 25                 | A+                 |
| Grace    | Finance      | 2019-08-30 | 68,000     | 10%     | True   | 14                 | A                  |
| Heidi    | HR           | 2022-02-14 | 47,000     | 3%      | True   | 5                  | B-                 |
| Ivan     | Engineering  | 2021-05-21 | 60,000     | 8%      | False  | 10                 | B+                 |
| Judy     | Marketing    | 2018-12-01 | 55,000     | 6%      | True   | 8                  | B                  |

"""

    doc = Document(page_content=sample_markdown, metadata={"source": "demo.md"})
    processed_docs = process_markdown_doc(doc, rows_per_chunk=3)
    index_documents(processed_docs)
    # Possible changes:
    #  - Add data schema directly in metadata
    #  - Ask LLM to provide a short summary for each table and store it

    #example_markdown = pathlib.Path("/home/phaman/Project-Master/TableSense-AI/tablesense_ai/agent/rag/sample_markdwon.md")
    #converter = Converter()

    #processed_docs = converter.convert(
    #    path_to_file=example_markdown,
    #    output_format=TableFormat.NATURAL,
    #    save_to_file=False)

    #doc = Document(page_content=sample_markdown, metadata={"source": "demo.md"})

    #processed_docs = process_markdown_doc(doc, rows_per_chunk=3, natural=True)
    #index_documents(processed_docs)
    for d in processed_docs:
        print("----")
        print(d.page_content)
        print(d.metadata)
