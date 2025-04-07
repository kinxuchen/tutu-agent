from fastapi import APIRouter
from components.db import Session
from pydantic import BaseModel, Field
import traceback
from typing import Any, List, Sequence
from services.inventory import insert_inventory as insert_inventory_db, insert_vector_inventory
from dto.inventory_dto import InventoryDTO, EmbeddingDTO, ClienteleDTO, ExampleListDTO
from services.clientele import insert_vector_clientele, insert_clientele
from services.example import insert_example_into_vector
from llm import embeddings

gpts_router = APIRouter(prefix='/api')


# 数据库中插入数据
@gpts_router.post('/inventory')
async def insert_inventory(body: InventoryDTO):
    try:
        # 插入对应数据，然后向量化这条数据
        inventor = insert_inventory_db(request=body)
        # 将数据插入到向量数据库中
        insert_vector_inventory(inventor)
        return {
            'message': '添加数据成功',
            'success': True
        }
    except Exception as e:
        return {
            'success': False,
            'message': str(e)
        }


# 添加客户信息到库中
@gpts_router.post('/add_clientele')
async def insert_clientele_request(body: ClienteleDTO):
    try:
        # todo 需要加上去重逻辑，此处先不处理
        client = insert_clientele(body)
        # 将数据添加到向量数据库中
        await insert_vector_clientele(client)
        return {
            'message': '新增客户成功',
            'success': True
        }
    except Exception as e:
        return {
            'success': False,
            'message': str(e)
        }


# 插入示例数据
@gpts_router.post('/add_examples')
async def insert_examples_request(body: List[dict]):
    try:
        await insert_example_into_vector(body)
        return {
            "success": True,
        }
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(error_traceback)
        return {
            "success": False,
            "message": str(e)
        }


# 新增一个接口，用于 embeddings 处理
@gpts_router.post('/embeddings')
async def embeddings_request(body: EmbeddingDTO):
    embedding = embeddings.embed_query(body.text)
    return embedding
