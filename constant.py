from dotenv import load_dotenv
import os
from langchain_community.utilities.sql_database import SQLDatabase

ENV = os.getenv('ENV', 'production')
# load_dotenv('.env' if ENV == 'development' else '.env.production')
if ENV == 'development':
    load_dotenv('.env')

MODEL_NAME=os.getenv('MODEL_NAME')
print('模型名称', MODEL_NAME)
API_KEY=os.getenv('API_KEY')
BASE_URL=os.getenv('BASE_URL')

# 视觉模型相关
IMAGE_MODEL = os.getenv('IMAGE_MODEL')
IMAGE_API_KEY = os.getenv('IMAGE_API_KEY')
IMAGE_BASE_URL=os.getenv('IMAGE_BASE_URL')


REDIS_HOST=os.getenv('REDIS_HOST')
REDIS_DB=os.getenv('REDIS_DB')
REDIS_PORT=os.getenv('REDIS_PORT')

MYSQL_HOST=os.getenv('MYSQL_HOST')
MYSQL_DB=os.getenv('MYSQL_DATABASE')
MYSQL_PORT=os.getenv('MYSQL_PORT')
MYSQL_USER=os.getenv('MYSQL_USER')
MYSQL_PASSWORD=os.getenv('MYSQL_PASSWORD')
SQL_URL = "mysql+pymysql://{}:{}@{}:{}/{}".format(MYSQL_USER,MYSQL_PASSWORD,MYSQL_HOST,MYSQL_PORT,MYSQL_DB)

MILVUS_HOST=os.getenv('MILVUS_HOST')
MILVUS_TOKEN=os.getenv('MILVUS_TOKEN')
MILVUS_PASSWORD=os.getenv('MILVUS_PASSWORD')
MILVUS_USER=os.getenv('MILVUS_USER')

EMBEDDING_API_KEY=os.getenv('EMBEDDING_API_KEY')
EMBEDDING_BASE_URL=os.getenv('EMBEDDING_BASE_URL')
EMBEDDING_MODEL=os.getenv('EMBEDDING_MODEL')

# 集合名称
COLLECTION_TUTU_NAME= 'tutu'
COLLECTION_EXAMPLE_NAME='example'
COLLECTION_RAG_NAME='rag' # 向量数据库名称
# 创建一个分区，用于存放客户数据
PARTITION_STORE_NAME='INVENTORY' # 库存分区
PARTITION_CLIENTELE_NAME='CLIENTELE' # 客户分区
PARTITION_EXAMPLE_NAME='EXAMPLE' # 示例数据

# 配置 oss 相关
OSS_ACCESS_KEY_ID=os.getenv('OSS_ACCESS_KEY_ID')
OSS_ACCESS_KEY_SECRET=os.getenv('OSS_ACCESS_KEY_SECRET')
OSS_ENDPOINT=os.getenv('OSS_ENDPOINT')
OSS_BUCKET_NAME=os.getenv('OSS_BUCKET_NAME')
OSS_REGION=os.getenv('OSS_REGION')
OSS_ALIAS_NAME=os.getenv('OSS_ALIAS_NAME')

COS_SECRET_ID = os.getenv('COS_SECRET_ID')
COS_SECRET_KEY = os.getenv('COS_SECRET_KEY')
COS_REGION = os.getenv('COS_REGION')
COS_BUCKET_NAME = os.getenv('COS_BUCKET_NAME')

# 豆包模型
DOUBAO_THINK_MODEL=os.getenv('DOUBAO_THINK_MODEL')
DOUBAO_THINK_BASE_URL=os.getenv('DOUBAO_THINK_BASE_URL')
DOUBAO_THINK_API_KEY=os.getenv('DOUBAO_THINK_API_KEY')

db = SQLDatabase.from_uri(SQL_URL)


db_schema = db.get_context()

print('数据库上下文', db_schema)
