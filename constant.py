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

db = SQLDatabase.from_uri(SQL_URL)

db_schema = db.get_context()

print('数据库上下文', db_schema)
