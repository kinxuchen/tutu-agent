from langchain_core.tools import tool
from langchain_core.documents import Document
from typing import List
from components.store import milvus_vector_rag_store

knowledge_retriever = milvus_vector_rag_store.as_retriever()
@tool
def knowledge_search_tool(query: str) -> List[Document]:
    """
    根据用户的输入，去知识库中查询相关数据
    :param query: 查询的参数 
    :return: List[Document]
    """
    milvus_search = knowledge_retriever.invoke(input=query)
    return milvus_search
