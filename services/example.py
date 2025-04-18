"""
示例接口，用于手动的传入示例
"""
from typing import List
from uuid import uuid4
from dto.chat_request_dto import ExampleRequestBody
from components.store import milvus_vector_store

async def insert_example_into_vector(examples: List[ExampleRequestBody]):
    texts = [f"输入{example.input}\n\n输出{example.output}" for example in examples]
    ids = []
    metadatas = []
    for example in examples:
        id = str(uuid4())
        ids.append(id)
        metadatas.append({
            'user_id': '123',
            'input': example.input,
            'output': example.output,
        })
    await milvus_vector_store.aadd_texts(
        metadatas=metadatas,
        texts=texts,
        ids=ids,
    )


