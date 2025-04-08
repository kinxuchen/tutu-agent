from dto.inventory_dto import ClienteleDTO
from uuid import uuid4
from components.db import Session
from llm import embeddings
from constant import COLLECTION_TUTU_NAME, PARTITION_CLIENTELE_NAME
from entity.clientele import Clientele, GenderEnum


def insert_clientele(client: ClienteleDTO) -> ClienteleDTO:
    """将客户信息插入到数据库中"""
    client_dict = client.model_dump()
    with Session() as session:
        client.id = str(uuid4())
        gender_enum = GenderEnum(client_dict['gender'])
        client_dict['id'] = client.id
        client_dict['gender'] = gender_enum
        new_client = Clientele(**client_dict)
        session.add(new_client)
        session.commit()
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
