from components.db import  metadata, Table, engine, Session, insert
from uuid import uuid4
from dto.inventory_dto import InventoryDTO
from components.store import get_vector_store
from constant import COLLECTION_INVENTORY_NAME
from llm import embeddings

# 插入数据
async def insert_inventory(request: InventoryDTO):
    request_dict = request.dict()
    primary_key = str(uuid4())
    try:
        with engine.connect() as connection:
            with connection.begin():
                inventory_table = Table(
                    'inventory',
                    metadata,
                    autoload_with=engine
                )
                request_dict['id'] = primary_key
                insert_stmt = insert(inventory_table).values(request_dict)
                connection.execute(insert_stmt)

        milvus_client = get_vector_store()
        row = {
            'vector': embeddings.embed_query(f"""
                商品名称:{request_dict['name']}
                颜色:{request_dict['color']}
            """),
            'id': primary_key,
            'primary_key': str(uuid4())
        }
        insert_result = milvus_client.insert(COLLECTION_INVENTORY_NAME, row)
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name='id',
        )
        return request
    except Exception as e:
        raise e
