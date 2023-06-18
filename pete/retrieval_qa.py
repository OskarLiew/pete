import logging
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma
from cache import CACHE_DIR

from config import CONFIG

LOGGER = logging.getLogger(__name__)
THIS_DIR = Path(__file__).parent
PERSIST_DIR = CACHE_DIR / "chroma"


def get_qa_chain() -> RetrievalQA:
    vectorstore = _init_vectorstore()

    return RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",  # Puts documents in context as they are
        retriever=vectorstore.as_retriever(),
    )


def _init_vectorstore() -> Chroma:
    embedding = OpenAIEmbeddings()

    # Load existing vectorstore
    if PERSIST_DIR.exists():
        return Chroma(
            persist_directory=str(PERSIST_DIR.resolve()), embedding_function=embedding
        )

    LOGGER.info("Initializing vectorstore...")
    texts, metadata = _load_and_split_markdown(CONFIG.vault)

    vectorstore = Chroma.from_texts(
        texts,
        metadatas=metadata,
        embedding=embedding,
        persist_directory=str(PERSIST_DIR.resolve()),
    )
    vectorstore.persist()
    return vectorstore


def _load_and_split_markdown(root_dir: Path) -> tuple[list[str], list[dict[str, str]]]:
    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
    )
    texts, metadata = [], []
    for note_path in root_dir.glob("**/*.md"):
        with open(note_path, "r", encoding="utf-8") as file:
            text = file.read()

        md_splits = text_splitter.split_text(text)

        # Add source to metadata
        for split in md_splits:
            split["metadata"]["source"] = str(note_path.relative_to(root_dir.parent))

        texts.extend([split["content"] for split in md_splits])
        metadata.extend([split["metadata"] for split in md_splits])
    return texts, metadata
