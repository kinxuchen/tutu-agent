"""
示例接口，用于手动的传入示例
"""
from typing import List
from uuid import uuid4
from dto.inventory_dto import ExampleDTO
from components.store import milvus_vector_store
from constant import COLLECTION_EXAMPLE_NAME, MILVUS_HOST, MILVUS_TOKEN, MILVUS_USER, MILVUS_PASSWORD
from llm import embeddings

async def insert_example_into_vector(examples: List[ExampleDTO]):
    texts = [f"输入{example['input']}\n\n输出{example['output']}" for example in examples]
    ids = []
    metadatas = []
    for example in examples:
        id = str(uuid4())
        ids.append(id)
        metadatas.append({
            'id': id,
            'user_id': '123',
            'input': example['input'],
            'output': example['output'],
        })
    await milvus_vector_store.aadd_texts(
        metadatas=metadatas,
        texts=texts,
        ids=ids,
    )


