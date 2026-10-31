
from pathlib import Path
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KB_FILE = DATA_DIR / "VoltKart_Support_Knowledge_Base.pdf"
INDEX_DIR = BASE_DIR / "faiss_index"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Load and clean the PDF
# --------------------------------------------------

def load_documents():
    if not KB_FILE.is_file():
        raise FileNotFoundError(
            f"Knowledge base PDF not found: {KB_FILE}"
        )

    pages = PyPDFLoader(str(KB_FILE)).load()
    cleaned_pages = []

    for page in pages:
        page.metadata["source"] = KB_FILE.name

        text = page.page_content
        text = text.replace("\u2022", "-")

        lines = []

        for line in text.splitlines():
            stripped = line.strip()

            if stripped == "VoltKart Customer Support Knowledge Base":
                continue

            if stripped.lower().startswith("page ") and stripped[5:].isdigit():
                continue

            lines.append(line)

        page.page_content = "\n".join(lines).strip()

        if len(page.page_content) > 30:
            cleaned_pages.append(page)

    if not cleaned_pages:
        raise ValueError("No readable text found in the VoltKart PDF.")

    logger.info("Loaded %d PDF pages", len(cleaned_pages))
    return cleaned_pages


# --------------------------------------------------
# Split documents into chunks
# --------------------------------------------------

def build_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata.setdefault("source", KB_FILE.name)

    logger.info("Created %d chunks", len(chunks))
    return chunks


# --------------------------------------------------
# Build and save the FAISS index
# --------------------------------------------------

def main():
    logger.info("Loading PDF...")
    pages = load_documents()

    logger.info("Splitting PDF into chunks...")
    chunks = build_chunks(pages)

    if not chunks:
        raise ValueError("No chunks were created from the PDF.")

    logger.info("Loading Hugging Face embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Creating FAISS vector index...")
    vector_store = FAISS.from_documents(
        chunks,
        embedding_model,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Saving FAISS index to: %s", INDEX_DIR)
    vector_store.save_local(str(INDEX_DIR))

    logger.info("FAISS index successfully created.")
    logger.info("Index files:")
    logger.info("  - %s", INDEX_DIR / "index.faiss")
    logger.info("  - %s", INDEX_DIR / "index.pkl")


if __name__ == "__main__":
    main()
