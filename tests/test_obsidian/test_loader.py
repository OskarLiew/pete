import re
from pete.chains.obsidian.loader import (
    MyObsidianLoader,
    _prune_obsidian_text,
    MyObsidianTextSplitter,
    _get_obsidian_document_title,
)
from langchain.docstore.document import Document

from pathlib import Path

TEST_NOTES_DIR = Path(__file__).parent / "notes"

loader = MyObsidianLoader(str(TEST_NOTES_DIR.resolve()))

md_text = """
# Title

This is a markdown file. Markdown is a markup language

---

# References

1. My brain

"""

md_text_pruned = """# Title

This is a markdown file. Markdown is a markup language

# References

1. My brain"""


document = Document(page_content=md_text_pruned, metadata={"source": "Title.md"})
document_alias = Document(
    page_content=md_text_pruned, metadata={"alias": "Title2", "source": "Title.md"}
)


def test_loader_e2e():
    documents = loader.load()
    assert len(documents) == 4


def test_prune_obsidian_text():
    pruned_text = _prune_obsidian_text(md_text)
    assert pruned_text == md_text_pruned


def test_get_obsidian_document_title_no_alias():
    title = _get_obsidian_document_title(document)
    assert title == "Title"


def test_get_obsidian_document_title_with_alias():
    title = _get_obsidian_document_title(document_alias)
    assert title == "Title, Title2"


def test_splitter():
    splitter = MyObsidianTextSplitter(chunk_size=80, chunk_overlap=0)
    split_documents = splitter.split_documents([document, document_alias])
    assert len(split_documents) > 2
    for doc in split_documents:
        title = _get_obsidian_document_title(doc)
        x = doc.page_content.startswith(
            f'Snippet from: {title}\n"""\n'
        ) and doc.page_content.endswith('\n"""')
        assert x
