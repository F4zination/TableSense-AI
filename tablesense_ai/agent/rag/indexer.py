import hashlib
import re
import pandas as pd
from io import StringIO
from langchain.docstore.document import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_huggingface import HuggingFaceEmbeddings
from tablesense_ai.agent.serialization.converter import Converter, TableFormat
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
import sqlite3
from sqlalchemy import create_engine
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

POSTGRES_URL = "postgresql+psycopg2://tablesense_user:tablesense_password@localhost:5433/tablesense_db"

def store_table_in_sql(df: pd.DataFrame, table_id: str, db_url: str = POSTGRES_URL):
    """
    Store a DataFrame as a SQL table with inferred dtypes for exact querying
    using SQLAlchemy and PostgreSQL.
    Table name = table_<table_id>.
    """
    try:
        engine = create_engine(db_url)
    except Exception as e:
        print(f"Failed to create SQLAlchemy engine with URL: {db_url}. Error: {e}")
        return

    table_name = f"table_{table_id}"

    df_clean = df.copy()
    df_clean.columns = [c.strip() for c in df_clean.columns]

    for col in df_clean.columns:
        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
        )
        df_clean[col] = pd.to_numeric(df_clean[col], errors="ignore")

    try:
        with engine.connect() as conn:
            df_clean.to_sql(table_name, conn, if_exists="replace", index=False)
        
        print(f"Successfully stored table in PostgreSQL as '{table_name}' ({len(df_clean)} rows)")
    except Exception as e:
        print(f"Failed to store table {table_id} in PostgreSQL: {e}")


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
            .dropna(axis=1, how="all")
            .iloc[1:]
        )
    except Exception as e:
        print(f"Failed to parse table: {e}")
        return None

def df_to_natural(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame into natural language sentences using Jinja2.
    """
    try:
        # This assumes 'jinja_templates' is in the same directory as this script
        # Create the directory if it doesn't exist
        template_dir = Path(__file__).resolve().parent / "jinja_templates"
        template_dir.mkdir(exist_ok=True)
        template_path = template_dir / "only_header.natural.jinja2"

        # Check if template file exists, if not, use fallback
        if not template_path.exists():
            print(f"Template not found at {template_path}, using fallback.")
            raise TemplateNotFound("Template file missing")

        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template('only_header.natural.jinja2')
        return template.render(dataframe=df)
    
    except Exception as e:
        if isinstance(e, TemplateNotFound):
             print(f"Template rendering failed ({e}), using fallback.")
        else:
             print(f"Template rendering failed with an unexpected error ({e}), using fallback.")

        # Fallback: describe first rows in simple sentences
        sentences = []
        for _, row in df.head(5).iterrows():
            parts = [f"{col} is '{row[col]}'" for col in df.columns]
            sentences.append(", ".join(parts))
        return "\n".join(sentences)


def chunk_table(df: pd.DataFrame, table_id: str, original_metadata: dict, rows_per_chunk: int = 5, natural: bool = False):
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

        # <--- CHANGED: Merge original_metadata to add the 'source' key
        metadata = {
            **original_metadata, 
            "table_id": f"table_{table_id}",
            "row_range": f"{start}-{start+len(chunk)-1}",
            "mode": "natural language don't execute code to parse it!!" if natural else "table"
        }
        documents.append(Document(page_content=chunk_text, metadata=metadata))
    return documents


def extract_and_chunk_free_text(doc: Document, table_matches: list, text_splitter: RecursiveCharacterTextSplitter):
    """
    Extracts free text from *around* the tables and splits it into chunks.
    """
    markdown_text = doc.page_content
    original_metadata = doc.metadata
    text_docs = []
    last_end = 0

    for _, start, end in table_matches:
        text_block = markdown_text[last_end:start].strip()
        if text_block:
            chunks = text_splitter.split_text(text_block)
            for chunk in chunks:
                # <--- CHANGED: Add empty keys to match the schema
                text_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        **original_metadata, 
                        "mode": "text",
                        "table_id": "",  # Ensures schema match
                        "row_range": ""  # Ensures schema match
                    } 
                ))
        last_end = end

    final_text_block = markdown_text[last_end:].strip()
    if final_text_block:
        chunks = text_splitter.split_text(final_text_block)
        for chunk in chunks:
            # <--- CHANGED: Add empty keys to match the schema
            text_docs.append(Document(
                page_content=chunk,
                metadata={
                    **original_metadata, 
                    "mode": "text",
                    "table_id": "",  # Ensures schema match
                    "row_range": ""  # Ensures schema match
                }
            ))
    
    if not table_matches and markdown_text.strip():
            chunks = text_splitter.split_text(markdown_text.strip())
            for chunk in chunks:
                # <--- CHANGED: Add empty keys to match the schema
                text_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        **original_metadata, 
                        "mode": "text",
                        "table_id": "",  # Ensures schema match
                        "row_range": ""  # Ensures schema match
                    }
                ))

    return text_docs


def process_markdown_doc(doc: Document, rows_per_chunk: int = 5, natural: bool = False):
    """
    Process a LangChain Document containing markdown text.
    Extracts tables, chunks them, and stores them (SQL + Vector).
    Also extracts and chunks the surrounding free text for vector indexing.
    """
    table_docs_out = []
    markdown_text = doc.page_content
    original_metadata = doc.metadata  # <--- CHANGED: Get original metadata
    tables = extract_markdown_tables(markdown_text)

    # 1. Process Tables
    for table_markdown, start, end in tables:
        table_id = hashlib.md5(table_markdown.encode()).hexdigest()[:10]
        df = markdown_table_to_df(table_markdown)
        if df is not None:
            store_table_in_sql(df, table_id) 
            # <--- CHANGED: Pass original_metadata so table chunks get the 'source' key
            table_docs_out.extend(chunk_table(df, table_id, original_metadata, rows_per_chunk, natural=natural))

    # 2. Process Free Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    text_docs_out = extract_and_chunk_free_text(doc, tables, text_splitter)

    # 3. Combine and return
    # We return text chunks first, but it doesn't matter
    # because both lists now have the same metadata keys.
    return table_docs_out + text_docs_out

def index_documents(docs: list[Document]):
    if not docs:
        print("No documents to index.")
        return
        
    print(f"\nAttempting to index {len(docs)} documents in Milvus...")
    try:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # --- Index in Milvus (Working) ---
        Milvus.from_documents(
            documents=docs,
            embedding=embedding,
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            connection_args={"uri": "http://localhost:19530"},
            consistency_level="Bounded",
            drop_old=True, # Deletes existing collection with the same name first
        )
        
        print("Successfully indexed documents in Milvus.")
    except Exception as e:
        print(f"Failed to index documents in Milvus: {e}")
        print("Please ensure Milvus is running at http://localhost:19530")


if __name__ == "__main__":

    
    file_path = Path("/home/deulai/table_agent/TableSense-AI/tablesense_ai/agent/rag/Q3_Business_Review.md")
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    doc = docs[0]
    doc.metadata["source"] = file_path.name
    #doc = Document(page_content=sample_markdown, metadata={"source": "demo.md"})
    
    # Process with rows_per_chunk=5
    processed_docs = process_markdown_doc(doc, rows_per_chunk=4, natural=False) 
    
    # --- This will now execute the actual indexing ---
    index_documents(processed_docs) 

    print("\n--- Generated Documents ---")
    for d in processed_docs:
        print("----")
        print(f"Metadata: {d.metadata}")
        print(d.page_content)
        
    print(f"\nTotal documents generated: {len(processed_docs)}")