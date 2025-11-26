import hashlib
import re
import pandas as pd
from io import StringIO
from pathlib import Path
from sqlalchemy import create_engine
from docling.document_converter import DocumentConverter

from langchain.docstore.document import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Jinja Imports not needed anymore, as we store tables in markdown
from jinja2 import Environment, FileSystemLoader, TemplateNotFound


POSTGRES_URL = "postgresql+psycopg2://tablesense_user:tablesense_password@localhost:5433/tablesense_db"

def store_table_in_sql(df: pd.DataFrame, table_id: str, db_url: str = POSTGRES_URL):
    """
    Store a DataFrame as a SQL table with inferred dtypes for exact querying.
    """
    try:
        engine = create_engine(db_url)
    except Exception as e:
        print(f"Failed to create SQLAlchemy engine: {e}")
        return

    table_name = f"table_{table_id}"

    df_clean = df.copy()
    # Clean column names
    df_clean.columns = [str(c).strip() for c in df_clean.columns]

    # Clean data
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
        print(f"Stored table in SQL: '{table_name}' ({len(df_clean)} rows)")
    except Exception as e:
        print(f"Failed to store table {table_id} in PostgreSQL: {e}")


def extract_markdown_tables(markdown_text: str):
    #This regex finds standard markdown tables
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

# Deprecated
def df_to_natural(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame into natural language sentences using Jinja2.
    """
    try:
        template_dir = Path(__file__).resolve().parent / "jinja_templates"
        template_dir.mkdir(exist_ok=True)
        template_path = template_dir / "only_header.natural.jinja2"

        if not template_path.exists():
            # Create a default template if missing
            with open(template_path, "w") as f:
                f.write("Columns: {{ dataframe.columns|join(', ') }}.\n")
                f.write("{% for i, row in dataframe.iterrows() %}")
                f.write("- Row {{ i }}: {{ row.to_dict() }}\n")
                f.write("{% endfor %}")

        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template('only_header.natural.jinja2')
        return template.render(dataframe=df)
    
    except Exception as e:
        print(f"Template error ({e}), using fallback.")
        sentences = []
        for _, row in df.head(5).iterrows():
            parts = [f"{col} is '{row[col]}'" for col in df.columns]
            sentences.append(", ".join(parts))
        return "\n".join(sentences)


def chunk_table(df: pd.DataFrame, table_id: str, original_metadata: dict, rows_per_chunk: int = 8, natural: bool = False):
    documents = []
    for start in range(0, len(df), rows_per_chunk):
        chunk = df.iloc[start:start+rows_per_chunk]

        if natural:
            chunk_text = df_to_natural(chunk)
        else:
            chunk_text = chunk.to_markdown(index=False)

        metadata = {
            **original_metadata, 
            "table_id": f"table_{table_id}",
            "row_range": f"{start}-{start+len(chunk)-1}",
            "mode": "natural" if natural else "table"
        }
        documents.append(Document(page_content=chunk_text, metadata=metadata))
    return documents


def extract_and_chunk_free_text(doc: Document, table_matches: list, text_splitter: RecursiveCharacterTextSplitter):
    markdown_text = doc.page_content
    original_metadata = doc.metadata
    text_docs = []
    last_end = 0

    # Helper to add doc
    def add_text_chunk(text):
        if text.strip():
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                text_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        **original_metadata, 
                        "mode": "text", 
                        "table_id": "", 
                        "row_range": ""
                    }
                ))

    for _, start, end in table_matches:
        text_block = markdown_text[last_end:start]
        add_text_chunk(text_block)
        last_end = end

    final_text_block = markdown_text[last_end:]
    add_text_chunk(final_text_block)
    
    return text_docs


def process_markdown_doc(doc: Document, rows_per_chunk: int = 5, natural: bool = False):
    """
    Process a LangChain Document (Markdown).
    Extracts tables -> SQL + Vector.
    Extracts text -> Vector.
    """
    table_docs_out = []
    markdown_text = doc.page_content
    original_metadata = doc.metadata
    
    # 1. Identify Tables
    tables = extract_markdown_tables(markdown_text)

    # 2. Process Tables
    for table_markdown, start, end in tables:
        # Create a unique ID based on content
        table_id = hashlib.md5(table_markdown.encode()).hexdigest()[:10]
        df = markdown_table_to_df(table_markdown)
        
        if df is not None and not df.empty:
            store_table_in_sql(df, table_id) 
            table_docs_out.extend(chunk_table(df, table_id, original_metadata, rows_per_chunk, natural=natural))

    # 3. Process Free Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    text_docs_out = extract_and_chunk_free_text(doc, tables, text_splitter)

    return table_docs_out + text_docs_out


def index_documents(docs: list[Document]):
    if not docs:
        print("No documents to index.")
        return
        
    print(f"\nIndexing {len(docs)} documents in Milvus...")
    try:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        Milvus.from_documents(
            documents=docs,
            embedding=embedding,
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            connection_args={"uri": "http://localhost:19530"},
            consistency_level="Bounded",
            drop_old=True, # --> remove for persistency
        )
        print("Successfully indexed documents in Milvus.")
    except Exception as e:
        print(f"Failed to index documents in Milvus: {e}")


def index_pdf(file_path: Path):
    """
    converts pdf to markdown using dockling and splits them up into
    langchain docs to index it
    """
    converter = DocumentConverter()
    result = converter.convert(file_path)
        
    markdown_content = result.document.export_to_markdown()
        
    # Create a LangChain Document from markdown
    doc = Document(
        page_content=markdown_content, 
        metadata={"source": file_path.name}
    )

    processed_docs = process_markdown_doc(doc, rows_per_chunk=5, natural=False) 
    
    index_documents(processed_docs) 

    print(f"\nTotal documents generated: {len(processed_docs)}")
