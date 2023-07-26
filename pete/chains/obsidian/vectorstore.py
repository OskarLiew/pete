from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from pete.chains.obsidian.loader import MyObsidianLoader, MyObsidianTextSplitter
from pete.cli_utils import progress_spinner
from pete.config import CONFIG

PERSIST_DIR = CONFIG.cache_path / "chroma"


def init_vectorstore() -> Chroma:
    embedding = OpenAIEmbeddings()  # type: ignore
    persist_dir = str(PERSIST_DIR.resolve())

    # Load existing vectorstore
    if PERSIST_DIR.exists():
        return Chroma(persist_directory=persist_dir, embedding_function=embedding)

    with progress_spinner("Loading documents..."):
        loader = MyObsidianLoader(str(CONFIG.vault))
        text_splitter = MyObsidianTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = loader.load_and_split(text_splitter)

    with progress_spinner("Embedding documents..."):
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_dir,
        )

    vectorstore.persist()
    return vectorstore
