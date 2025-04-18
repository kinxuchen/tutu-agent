from fastapi import FastAPI
from langgraph.graph.state import CompiledStateGraph
from components.db import init_db, close_db
from components.store import initial_vector_collection, close_vector_store
from apis.gpts import gpts_router
from apis.agent import agent_router
from apis.rag import rag_router
from apis.tools import tool_router
from contextlib import asynccontextmanager


redis = None
redis_async = None
agent: CompiledStateGraph | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    initial_vector_collection()
    init_db()
    yield
    close_vector_store()
    close_db()

app = FastAPI(lifespan=lifespan)

app.include_router(gpts_router)
app.include_router(agent_router)
app.include_router(tool_router)
app.include_router(rag_router)

# 启动服务连接 Redis 和创建智能体



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
