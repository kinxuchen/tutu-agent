from dto.inventory_dto import ClienteleDTO
from uuid import uuid4
from components.db import engine, metadata, Table,insert, Session
from llm import embeddings
from constant import COLLECTION_TUTU_NAME, PARTITION_CLIENTELE_NAME


def insert_clientele(client: ClienteleDTO) -> ClienteleDTO:
    """将客户信息插入到数据库中"""
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


async def insert_vector_clientele(client: ClienteleDTO):
    """将客户信息添加到向量数据库中"""
    from components.store import get_vector_store
    vector_store = get_vector_store()
    vector = embeddings.embed_query(client.name)
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
