import threading
from typing import Literal
from agents.main_agent import create_agent
from constant import REDIS_DB, REDIS_PORT, REDIS_HOST
from redis.asyncio import Redis as RedisAsync
from redis import Redis
class Agent:
    _instance = None
    _agent = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance:
            return cls._instance
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def get_agent(self):
        if not self._instance._agent:
            from components.store import redis
            self._instance._agent = create_agent(redis)
        return self._agent

