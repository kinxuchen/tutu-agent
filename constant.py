from dotenv import load_dotenv
import os
from langchain_community.utilities.sql_database import SQLDatabase

env_path = os.path.join(os.getcwd(), '.env')

load_dotenv(env_path)

MODEL_NAME=os.getenv('MODEL_NAME')
API_KEY=os.getenv('API_KEY')
BASE_URL=os.getenv('BASE_URL')


REDIS_HOST=os.getenv('REDIS_HOST')
REDIS_DB=os.getenv('REDIS_DB')
REDIS_PORT=os.getenv('REDIS_PORT')

MYSQL_HOST=os.getenv('MYSQL_HOST')
MYSQL_DB=os.getenv('MYSQL_DB')
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

COLLECTION_INVENTORY_NAME='inventory'

db = SQLDatabase.from_uri(SQL_URL)


db_schema = db.get_context()

print('数据库上下文', db_schema)
