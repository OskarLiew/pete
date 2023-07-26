from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

from pete.chains.obsidian.vectorstore import init_vectorstore


def init_obsidian_retriever():
    vectorstore = init_vectorstore()
    return MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(k=2), llm=ChatOpenAI(temperature=0)
    )
