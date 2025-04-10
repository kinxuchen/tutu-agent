from fastapi import FastAPI
from langgraph.graph.state import CompiledStateGraph
from components.db import init_db, close_db
from components.store import initial_vector_collection, close_vector_store
from agents.agent import Agent
from apis.gpts import gpts_router
from apis.agent import agent_router
from contextlib import asynccontextmanager


redis = None
redis_async = None
agent: CompiledStateGraph | None = None

app = FastAPI()

app.include_router(gpts_router)
app.include_router(agent_router)

# 启动服务连接 Redis 和创建智能体
@asynccontextmanager
async def lifespan(app: FastAPI):
    initial_vector_collection()
    init_db()
    Agent()
    yield
    close_vector_store()
    close_db()
    


@app.get('/')
async def root():
    return {"message": "Hello World"}
# 插入数据

if __name__ == "__main__":
    import uvicorn
    import os

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host=host, port=port)
