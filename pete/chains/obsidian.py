import logging
from pathlib import Path
from typing import Iterable, List

from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.document_loaders import ObsidianLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

from pete.cli_utils import progress_spinner
from pete.config import CONFIG

LOGGER = logging.getLogger(__name__)
THIS_DIR = Path(__file__).parent
PERSIST_DIR = CONFIG.cache_path / "chroma"

OBSIDIAN_NOTE_PROMPT = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Note source file: {source}\n\n{page_content}",
)


class MyObsidianLoader(ObsidianLoader):
    """Obsidian dataloader that adds the note file stub and aliases as H1,
    because I do not use H1 in my notes. Ignores files that start with a header.
    """

    def load(self) -> List[Document]:
        documents = super().load()
        for document in documents:
            if self._remove_front_matter(document.page_content)[0] == "#":
                continue

            note_title = _get_obsidian_document_title(document)
            new_content = f"# {note_title}\n\n{document.page_content}"
            document.page_content = new_content
        return documents


class MyObsidianTextSplitter(MarkdownTextSplitter):
    """Text splitter that adds note title and aliases to each document so
    it's possible to query specific notes.
    """

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        documents = super().split_documents(documents)
        for document in documents:
            note_title = _get_obsidian_document_title(document)
            new_content = f"Note title: {note_title}\n\n{document.page_content}\n\nEnd of: {note_title}"
            document.page_content = new_content
        return documents


def _get_obsidian_document_title(document: Document):
    note_title = document.metadata["source"].replace(".md", "")
    aliases = document.metadata.get("alias") or document.metadata.get("aliases")
    return f"{note_title}" + (f", {aliases}" or "")


def get_obsidian_chain() -> RetrievalQA:
    vectorstore = _init_vectorstore()

    return RetrievalQA.from_chain_type(  # type: ignore
        llm=OpenAI(),  # type: ignore
        chain_type="stuff",  # Puts documents in context as they are
        retriever=vectorstore.as_retriever(),
        # chain_type_kwargs={"document_prompt": None},  # BasePromptTemplate
    )


def _init_vectorstore() -> Chroma:
    embedding = OpenAIEmbeddings()  # type: ignore
    persist_dir = str(PERSIST_DIR.resolve())

    # Load existing vectorstore
    if PERSIST_DIR.exists():
        return Chroma(persist_directory=persist_dir, embedding_function=embedding)

    with progress_spinner("Loading documents..."):
        loader = MyObsidianLoader(str(CONFIG.vault))
        text_splitter = MyObsidianTextSplitter(chunk_size=500, chunk_overlap=0)
        documents = loader.load_and_split(text_splitter)

    with progress_spinner("Embedding documents..."):
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_dir,
        )

    vectorstore.persist()
    return vectorstore
