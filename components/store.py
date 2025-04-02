from pymilvus import MilvusClient
import threading
from constant import MILVUS_HOST, MILVUS_TOKEN, MILVUS_USER, MILVUS_PASSWORD

_milvus_client = None


# 获取向量数据库
def get_vector_store():
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient(
            uri=MILVUS_HOST,
            token=MILVUS_TOKEN,
            user=MILVUS_USER,
            password=MILVUS_PASSWORD
        )
    return _milvus_client
