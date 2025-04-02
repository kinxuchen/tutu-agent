from fastapi import FastAPI
from langgraph.graph.state import CompiledStateGraph
from components.db import init_db
from agents.agent import Agent
from apis.gpts import gpts_router
from apis.agent import agent_router

redis = None
redis_async = None
agent: CompiledStateGraph | None = None

app = FastAPI()

app.include_router(gpts_router)
app.include_router(agent_router)


# 启动服务连接 Redis 和创建智能体
@app.on_event('startup')
async def startup():
    init_db()
    Agent()

# 插入数据

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8002)
