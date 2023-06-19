import logging
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.document_loaders import ObsidianLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Chroma

from pete.cli_utils import progress_spinner
from pete.config import CONFIG

LOGGER = logging.getLogger(__name__)
THIS_DIR = Path(__file__).parent
PERSIST_DIR = CONFIG.cache_path / "chroma"


def get_qa_chain() -> RetrievalQA:
    vectorstore = _init_vectorstore()

    return RetrievalQA.from_chain_type(  # type: ignore
        llm=OpenAI(),  # type: ignore
        chain_type="stuff",  # Puts documents in context as they are
        retriever=vectorstore.as_retriever(),
    )


def _init_vectorstore() -> Chroma:
    embedding = OpenAIEmbeddings()  # type: ignore
    persist_dir = str(PERSIST_DIR.resolve())

    # Load existing vectorstore
    if PERSIST_DIR.exists():
        return Chroma(persist_directory=persist_dir, embedding_function=embedding)

    with progress_spinner("Loading documents..."):
        loader = ObsidianLoader(str(CONFIG.vault))
        text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = loader.load_and_split(text_splitter)

    with progress_spinner("Embedding documents..."):
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_dir,
        )

    vectorstore.persist()
    return vectorstore
