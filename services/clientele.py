from dto.inventory_dto import ClienteleDTO
from uuid import uuid4
from components.db import engine, metadata, Table,insert, Session
from llm import embeddings
from constant import COLLECTION_TUTU_NAME, PARTITION_CLIENTELE_NAME


# 数据库中插入数据
def insert_clientele(client: ClienteleDTO) -> ClienteleDTO:
    client_dict = client.model_dump()
    with Session() as session:
        clientele_table = Table(
            'clients',
            metadata,
            autoload_with=engine
        )
        client.id = str(uuid4())
        client_dict['id'] = client.id
        insert_stmt = insert(clientele_table).values(client_dict)
        session.execute(insert_stmt)
    return client


# 在向量数据库中插入数据
async def insert_vector_clientele(client: ClienteleDTO):
    from components.store import get_vector_store
    vector_store = get_vector_store()
    vector_text = f"""
        客户姓名：{client.name};
        客户性别：{'男性' if client.gender == 'male' else '女性'};
        客户年龄：{client.age};
    """
    vector = embeddings.embed_query(vector_text)
    row = {
        'vector': vector,
        'id': str(uuid4()),
        'user_id': '123',
        'metadata': client.model_dump()
    }
    vector_store.insert(
        collection_name=COLLECTION_TUTU_NAME,
        data=row,
        partition_name=PARTITION_CLIENTELE_NAME
    )
    vector_store.flush(collection_name=COLLECTION_TUTU_NAME)
