import jsonpickle
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.graph.state import CompiledStateGraph
from agents.main_agent import  create_agent, MainAgentState
from redis.asyncio import Redis as RedisAsync
from redis import Redis
import traceback
from langchain_core.messages import AIMessage, AIMessageChunk
from constant import REDIS_DB, REDIS_PORT, REDIS_HOST
from pydantic import BaseModel
from threading import Thread
from langchain_core.messages import BaseMessage, HumanMessage

redis = None
redis_async = None
agent: CompiledStateGraph | None = None

app = FastAPI()

class ChatRequestBody(BaseModel):
    input: str # 用户输入
    is_create_order: bool = False # 是否需要创建订单


# 启动服务连接 Redis 和创建智能体
@app.on_event('startup')
async def startup():
    global redis
    global redis_async
    global agent
    redis = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    redis_async = RedisAsync(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    agent = create_agent(redis_async=redis_async, redis=redis)

@app.post('/chat/{user_id}/{thread_id}')
async def chat(user_id: str, thread_id: str, body: ChatRequestBody):
    print('userid', user_id)
    print('thread_id', thread_id)
    if agent is None:
        return {"success": False }
    try:
        config = {
            "configurable": {
                "thread_id": "session_id",
                "user_id": "user_id"
            }
        }
        agent_state = agent.get_state(config=config)
        messages = agent_state.values.get('messages', [])
        result = agent.invoke(input={
            "messages": messages + [HumanMessage(content=body.input)],
            'is_create_order': body.is_create_order
        }, config=config)
        print('输出结果', result)
        return {
            "success": True
        }
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(error_traceback)
        return {
            "success": False
        }


# 流式响应接口
# 需要考虑，单据接口是否需要保存历史记录？
# 普通 AI 会话聊天是否保存历史记录
@app.post('/stream/{user_id}/{thread_id}')
async def stream(user_id: str, thread_id: str, body: ChatRequestBody):
    if agent is None:
        return {"success": False }
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id
        }
    }
    agent_state = agent.get_state(config=config)
    def generator_stream():
        for (chunk, message) in agent.stream(input={
            "messages": [HumanMessage(content=body.input)],
            'is_create_order': False
        }, config=config, stream_mode="messages"):
            if isinstance(chunk, AIMessageChunk):
                print(chunk)
                yield f'data:{chunk.content}\n\n'
        yield f'data:{jsonpickle.encode({"status": "completed"})}'

    return StreamingResponse(
        generator_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
@app.get('/')
def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8002)
