from sqlalchemy import create_engine, MetaData, Table, insert
from sqlalchemy.ext.declarative import declarative_base
from constant import MYSQL_HOST, MYSQL_DB, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD
from sqlalchemy.orm import sessionmaker

SQL_URL = "mysql+pymysql://{}:{}@{}:{}/{}".format(MYSQL_USER,MYSQL_PASSWORD,MYSQL_HOST,MYSQL_PORT,MYSQL_DB)

engine = create_engine(SQL_URL)

print('连接')

metadata = MetaData()

Base = declarative_base()

Session: sessionmaker = sessionmaker(bind=engine, autoflush=True)
# 数据库初始化
def init_db():
    Base.metadata.create_all(bind=engine)

