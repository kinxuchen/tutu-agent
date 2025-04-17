from fastapi import UploadFile
from typing import List
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from components.store import milvus_vector_rag_store
import tempfile
import os

def add_metadata_field(x):
    x.metadata['url'] = x.metadata['source']
    return x

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=10
)
async def reader_markdown_content(files: List[UploadFile]):
    """加载 Markdown 格式并处理"""
    try:
        for file in files:
            file_name = file.filename
            if not file_name.endswith('.md'):
                continue
            file_content = await file.read()
            with tempfile.NamedTemporaryFile(delete=True, mode='wb', suffix='.md') as temp_file:
                temp_file.write(file_content)
                temp_file.flush()
                temp_file_path = temp_file.name
                loader = UnstructuredMarkdownLoader(
                    file_path=temp_file_path,
                    mode='single',
                    strategy='fast'
                )
                # 加载 Markdown 文档
                docs = loader.load()
                # 对文档进行切割
                documents = text_splitter.split_documents(docs)
                # 将文档数据上传到向量数据库
                milvus_vector_rag_store.add_documents(documents=list(map(add_metadata_field, documents)))
    except Exception as e:
        return []

async def query_knowledge_server(input: str):
    retriever = milvus_vector_rag_store.as_retriever()
    result = await retriever.ainvoke(input=input)
    return result

