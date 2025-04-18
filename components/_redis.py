from constant import REDIS_HOST, REDIS_PORT, REDIS_DB
from redis import Redis,ConnectionPool

pool = ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
redis = Redis(connection_pool=pool)
