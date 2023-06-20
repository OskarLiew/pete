import logging
import re
from pathlib import Path
from typing import Iterable, List

from langchain.docstore.document import Document
from langchain.document_loaders import ObsidianLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import MarkdownTextSplitter

from pete.config import CONFIG

LOGGER = logging.getLogger(__name__)
THIS_DIR = Path(__file__).parent
PERSIST_DIR = CONFIG.cache_path / "chroma"

OBSIDIAN_NOTE_PROMPT = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Note source file: {source}\n\n{page_content}",
)

SEPARATOR_PATTERN = re.compile(r"\n---\n")


class MyObsidianLoader(ObsidianLoader):
    """Obsidian dataloader that adds the note file stub and aliases as H1,
    because I do not use H1 in my notes. Does nothing for files that start with H1.
    """

    def load(self) -> List[Document]:
        documents = super().load()
        for document in documents:
            document.page_content = _prune_obsidian_text(document.page_content)
            if document.page_content.startswith("# "):
                continue

            note_title = _get_obsidian_document_title(document)
            new_content = f"# {note_title}\n\n{document.page_content}"
            document.page_content = new_content
        return documents


def _prune_obsidian_text(text: str) -> str:
    new_text = text.strip(" \n")
    new_text = SEPARATOR_PATTERN.sub("", new_text)
    return new_text


class MyObsidianTextSplitter(MarkdownTextSplitter):
    """Text splitter that adds note title and aliases to each document so
    it's possible to query specific notes.
    """

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        documents = super().split_documents(documents)
        for document in documents:
            note_title = _get_obsidian_document_title(document)
            new_content = (
                f'Snippet from: {note_title}\n"""\n{document.page_content}\n"""'
            )
            document.page_content = new_content
        return documents


def _get_obsidian_document_title(document: Document):
    note_title = document.metadata["source"].replace(".md", "")
    aliases = document.metadata.get("alias") or document.metadata.get("aliases")
    if aliases:
        note_title = f"{note_title}, {aliases}"
    return note_title
