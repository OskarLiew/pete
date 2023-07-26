import logging
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from pete.chains.obsidian.retriever import init_obsidian_retriever

LOGGER = logging.getLogger(__name__)
THIS_DIR = Path(__file__).parent


def init_obsidian_chain() -> RetrievalQA:
    retriever = init_obsidian_retriever()
    return RetrievalQA.from_chain_type(  # type: ignore
        llm=OpenAI(),  # type: ignore
        chain_type="stuff",  # Puts documents in context as they are
        retriever=retriever,
    )
