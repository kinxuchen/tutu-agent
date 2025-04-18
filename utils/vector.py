from components.store import get_vector_store
from llm import embeddings
from typing import Union, List

def vector_search(query: Union[str, List[str]], collection_name: str, partition_names:Union[List[str], None] = None, **kwargs):
    """
    向量检索
    :param query: 查询的文本
    :param collection_name: 需要查询的集合名称
    :param partition_name:  需要查询的分区名称
    :param kwargs: 其他 milvus 查询字段
    :return:
    """
    try:
        data = [embeddings.embed_query(query)] if isinstance(query, str) else list(map(lambda x: embeddings.embed_query(x), query))
        vector = get_vector_store()
        vector_result = vector.search(
            data=data,
            collection_name=collection_name,
            anns_field='vector',
            partition_names=partition_names if partition_names else None,
            limit= kwargs.get('limit') if kwargs.get('limit') else 1,
            output_fields=kwargs.get('output_fields', ['id', 'db_id', 'metadata']) if kwargs.get('output_fields') else None,
            search_params=kwargs.get('search_params', {}) if kwargs.get('search_params') else {
                "params": {
                    'radius': 0.7,
                    'level': 5,
                    'range_filter': 1,
                    'enable_recall_calculation': True
                }
            }
        )
        return None if  len(vector_result) == 0 else vector_result
    except Exception as e:
        return None

