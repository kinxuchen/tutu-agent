from fastapi import APIRouter
from components.store import get_vector_store
from components.db import Session
from pydantic import BaseModel, Field
from typing import Any
from services.inventory import insert_inventory as insert_inventory_db
from dto.inventory_dto import  InventoryDTO


gpts_router = APIRouter()


# 数据库中插入数据
@gpts_router.post('/api/inventory')
async def insert_inventory(request: InventoryDTO):
    try:
        # 插入对应数据，然后向量化这条数据
        await insert_inventory_db(request=request)
        return {
            'message': '添加数据成功',
            'success': True
        }
    except Exception as e:
        return {
            'success': False,
            'message': str(e)
        }
