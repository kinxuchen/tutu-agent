import uuid
from langchain_core.tools import tool
from agents.receipt.receipt_dto import BaseGoodResult
from typing import List
from jsonpickle import encode
from components.store import get_vector_store
from llm import embeddings
from constant import COLLECTION_TUTU_NAME, PARTITION_STORE_NAME, PARTITION_CLIENTELE_NAME
from pydash import get
from uuid import uuid4

@tool()
def vecotr_search(order_info: List[BaseGoodResult]):
    """
    从向量数据库中提取用户问题中的客户信息和库存信息
    Args:
    order_info: (List[GoodResult]) 一个包含商品名称，商品颜色，商品数量以及客户信息的信息的列表
    """
    vector_store =get_vector_store()
    search_data = [{
        'name': None if len(x['sku_result']) == 0 else get(x, ['sku_result', 0, 0, 'entity', 'metadata', 'name'], default=None),
        'good_id': None if len(x['sku_result']) == 0 else get(x, ['sku_result', 0, 0, 'entity', 'metadata', 'id'], default=None),
        'color': None if len(x['sku_result']) == 0 else get(x, ['sku_result', 0, 0, 'entity', 'metadata', 'color'], default=None),
        'clientele': None if len(x['clientele_result']) == 0 else get(x, ['clientele_result', 0, 0, 'entity', 'metadata', 'name'], default=None),
        'clientele_id': None if len(x['clientele_result']) == 0 else get(x, ['clientele_result', 0, 0, 'entity', 'metadata', 'id'], default=None),
        'count': get(x, ['count'], 0),
        'task_id': str(uuid4()), # 生成一个当前任务的 id，提供给后续操作使用
    } for x in [
        {
            'sku_result': vector_store.search(
                data=[embeddings.embed_query(
                    f"""商品名称: {good.name}\n商品颜色: {good.color if good.color is not None else ''}"""
                )] if (good.name is not None or good.color != '') else [],
                collection_name=COLLECTION_TUTU_NAME,
                anns_field='vector',  # 需要查询的字段
                partition_names=[PARTITION_STORE_NAME],
                limit=1,
                output_fields=['id', 'db_id', 'metadata'],
                search_params={
                    "params": {
                        'radius': 0.7,
                        'level': 5,
                        'range_filter': 1,
                        'enable_recall_calculation': True
                    }
                }
            ),
            'clientele_result': [] if (good.clientele is None or good.clientele == '') else vector_store.search(
                data=[embeddings.embed_query(
                    f"""{good.clientele}"""
                )],
                collection_name=COLLECTION_TUTU_NAME,
                anns_field='vector',  # 需要查询的字段
                partition_names=[PARTITION_CLIENTELE_NAME],
                limit=1,
                output_fields=['id', 'db_id', 'metadata'],
                search_params={
                    "params": {
                        'radius': 0.7,
                        'level': 5,
                        'enable_recall_calculation': True
                    }
                }
            ),
            'name': good.name,
            'color': good.color,
            'clientele': good.clientele,
            'count': good.count
        } for good in filter(lambda x: x.name is not None, order_info)
    ]]
    return search_data


@tool()
def clientele_vector_search(clientele_name: str):
    """
    请你根据用户输入的客户姓名，从向量数据库中提取客户信息
    Args: clientele_name (str): 客户姓名
    """
    vector_store =get_vector_store()
    clientele = vector_store.search(
        data=[embeddings.embed_query(
            clientele_name
        )],
        collection_name=COLLECTION_TUTU_NAME,
        anns_field='vector',  # 需要查询的字段
        partition_names=[PARTITION_CLIENTELE_NAME],
        limit=1,
        output_fields=['id', 'db_id', 'metadata'],
        search_params={
            "params": {
                'radius': 0.7,
                'level': 5,
                'range_filter': 1,
                'enable_recall_calculation': True
            }
        }
    )
    if len(clientele) == 0:
        return None
    return {
        'clientele_id': get(clientele, [0, 0, 'entity', 'metadata', 'id']),
        'clientele': get(clientele, [0, 0, 'entity', 'metadata', 'name']),
    }

