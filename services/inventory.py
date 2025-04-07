from components.db import  metadata, Table, engine, insert, Session
from uuid import uuid4
from dto.inventory_dto import InventoryDTO
from constant import COLLECTION_TUTU_NAME, PARTITION_STORE_NAME
from llm import embeddings


# 检查表并创建向量数据库
# 插入数据
def insert_inventory(request: InventoryDTO):
    request_dict = request.model_dump()
    primary_key = str(uuid4())
    request.id = primary_key
    with Session() as session:
        inventory_table = Table(
            'inventory',
            metadata,
            autoload_with=engine
        )
        request_dict['id'] = primary_key
        insert_stmt = insert(inventory_table).values(request_dict)
        session.execute(insert_stmt)
    return request


# 插入库存数据到向量数据库中
def insert_vector_inventory(request: InventoryDTO):
    from components.store import get_vector_store
    request_dict = request.model_dump()
    milvus_client = get_vector_store()
    row = {
        'vector': embeddings.embed_query(f"""商品名称:{request_dict['name']}\n商品颜色:{request_dict['color']}
            """),
        'id': str(uuid4()),
        'metadata': request_dict,
        'user_id': '123' # 模拟一个用户 id
    }
    milvus_client.insert(
        collection_name=COLLECTION_TUTU_NAME,
        data=row,
        partition_name=PARTITION_STORE_NAME
    )
    milvus_client.flush(collection_name=COLLECTION_TUTU_NAME)
