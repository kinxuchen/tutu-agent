from pymilvus import MilvusClient, DataType
import threading
from constant import (
    MILVUS_HOST,
    MILVUS_TOKEN,
    MILVUS_USER,
    MILVUS_PASSWORD,
    COLLECTION_TUTU_NAME,
    PARTITION_STORE_NAME,
    PARTITION_CLIENTELE_NAME,
    PARTITION_EXAMPLE_NAME,
    COLLECTION_EXAMPLE_NAME,
    COLLECTION_RAG_NAME
)
from llm import embeddings
from langchain_milvus import Zilliz

lock = threading.Lock()
_milvus_client = None


# 获取向量数据库
def get_vector_store():
    with lock:
        global _milvus_client
        if _milvus_client is None:
            _milvus_client = MilvusClient(
                uri=MILVUS_HOST,
                token=MILVUS_TOKEN,
                user=MILVUS_USER,
                password=MILVUS_PASSWORD
            )
    return _milvus_client

# 关闭向量数据库
def close_vector_store():
    with lock:
        global _milvus_client
        if _milvus_client is not None:
            _milvus_client.close()
            _milvus_client = None

# 初始化向量数据库
def initial_vector_collection():
    vectory_store = get_vector_store()
    is_tutu_collection = vectory_store.has_collection(COLLECTION_TUTU_NAME)
    is_example_collection = vectory_store.has_collection(COLLECTION_EXAMPLE_NAME)
    # 如果不存在，则创建
    if not is_tutu_collection:
        tutu_schema = vectory_store.create_schema(enable_dynamic_field=True)
        tutu_schema.add_field('vector', DataType.FLOAT_VECTOR, dim=1024, description="向量数据")
        tutu_schema.add_field('id', DataType.VARCHAR, is_primary=True, max_length=36)
        # 添加用户id
        tutu_schema.add_field(
            'user_id',
            DataType.VARCHAR,
            max_length=36,
            description="用户id",
        )
        tutu_schema.add_field(
            field_name='metadata',
            datatype=DataType.JSON,
            nullable=True,
            description="添加字段元数据"
        )
        index_params = vectory_store.prepare_index_params()
        index_params.add_index(
            field_name='vector',
            metric_type='COSINE',
            index_type="AUTOINDEX",
            index_name='vector_index'
        )
        vectory_store.create_collection(
            collection_name=COLLECTION_TUTU_NAME,
            schema=tutu_schema,
            primary_field_name='id',
            index_params=index_params
        )

    if not is_example_collection:
        example_schema = vectory_store.create_schema(enable_dynamic_field=True)
        example_schema.add_field('vector', DataType.FLOAT_VECTOR, dim=1024, description="向量数据")
        example_schema.add_field('id', DataType.VARCHAR, max_length=36, is_primary=True)
        # 添加用户id
        example_schema.add_field(
            'user_id',
            DataType.VARCHAR,
            max_length=36,
            description="用户id",
        )
        example_schema.add_field(
            'input',
            datatype=DataType.VARCHAR,
            max_length=65535,
            description="输入",
        )
        example_schema.add_field(
            'output',
            datatype=DataType.VARCHAR,
            max_length=65535,
            description="输出结果",
        )
        example_schema.add_field(
            'text',
            datatype=DataType.VARCHAR,
            max_length=65535,
            description="作为默认向量数据库存放原文本的数据",
        )
        example_index = vectory_store.prepare_index_params()
        example_index.add_index(
            field_name='vector',
            index_type='AUTOINDEX',
            metric_type='COSINE',
            index_name='vector_index'
        )
        vectory_store.create_collection(
            schema=example_schema,
            collection_name=COLLECTION_EXAMPLE_NAME,
            primary_field_name='id',
            consistency_level="Session",
            index_params=example_index,
        )



    # 添加库存分区
    if not vectory_store.has_partition(collection_name=COLLECTION_TUTU_NAME, partition_name=PARTITION_STORE_NAME):
        vectory_store.create_partition(
            collection_name=COLLECTION_TUTU_NAME,
            partition_name=PARTITION_STORE_NAME,
        )
    # 添加客户分区
    if not vectory_store.has_partition(collection_name=COLLECTION_TUTU_NAME, partition_name=PARTITION_CLIENTELE_NAME):
        vectory_store.create_partition(
            collection_name=COLLECTION_TUTU_NAME,
            partition_name=PARTITION_CLIENTELE_NAME
        )
    # 添加示例数据分区
    if not vectory_store.has_partition(collection_name=COLLECTION_TUTU_NAME, partition_name=PARTITION_EXAMPLE_NAME):
        vectory_store.create_partition(
            collection_name=COLLECTION_TUTU_NAME,
            partition_name=PARTITION_EXAMPLE_NAME
        )

# 使用 Zilliz 连接示例数据库
milvus_vector_store = Zilliz(
    collection_name=COLLECTION_EXAMPLE_NAME,
    embedding_function=embeddings,
    connection_args={
        'uri': MILVUS_HOST,
        'token': MILVUS_TOKEN,
        'user': MILVUS_USER,
        'password': MILVUS_PASSWORD
    },
    enable_dynamic_field=True,
    text_field='text',
    primary_field='id',
    auto_id=False,
    consistency_level="Session",
    index_params=[{
        'field_name': 'vector',
        'index_type': 'AUTOINDEX',
        'metric_type': 'COSINE',
    }]
)

milvus_vector_rag_store = Zilliz(
    collection_name=COLLECTION_RAG_NAME,
    embedding_function=embeddings,
    consistency_level="Session", # 一致性
    auto_id=True,
    vector_field="vector",
    text_field="text",
    connection_args={
        'uri': MILVUS_HOST,
        'token': MILVUS_TOKEN,
        'user': MILVUS_USER,
        'password': MILVUS_PASSWORD
    },
)
