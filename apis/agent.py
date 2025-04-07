from dto.chat_request_dto import ChatRequestBody
from fastapi import APIRouter
from agents.agent import Agent
from langchain_core.messages import HumanMessage
import traceback
from agents.receipt.agent import receipt_agent, ReceiptState

agent_router = APIRouter(prefix="/agent")

agent_instance = Agent()
@agent_router.post('/chat/{user_id}/{thread_id}')
async def chat(user_id: str, thread_id: str, body: ChatRequestBody):
    agent = agent_instance.get_agent()
    if agent is None:
        return {"success": False }
    try:
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id
            }
        }
        agent_state = agent.get_state(config=config)
        messages = agent_state.values.get('messages', [])
        result = agent.invoke(input={
            "messages": messages + [HumanMessage(content=body.input)],
            'is_create_order': body.is_create_order
        }, config=config)
        return {
            "success": True
        }
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(error_traceback)
        return {
            "success": False
        }

# 单据接口
@agent_router.post('/receipt/{user_id}/{thread_id}')
async def receipt_agent_request(user_id:str, thread_id: str, body: ChatRequestBody):
    config = {
        'configurable': {
            'thread_id': thread_id,
            'user_id': user_id,
        }
    }
    result = receipt_agent.invoke(input=ReceiptState(messages=[HumanMessage(content=body.input)]), config=config)
    return {
        'success': True,
        'result': result
    }

# 流式响应接口
# 需要考虑，单据接口是否需要保存历史记录？
# 普通 AI 会话聊天是否保存历史记录
# @agent_router.post('/stream/{user_id}/{thread_id}')
# async def stream(user_id: str, thread_id: str, body: ChatRequestBody):
#     if agent is None:
#         return {"success": False }
#     config = {
#         "configurable": {
#             "thread_id": thread_id,
#             "user_id": user_id
#         }
#     }
#     agent_state = agent.get_state(config=config)
#     def generator_stream():
#         for (chunk, message) in agent.stream(input={
#             "messages": [HumanMessage(content=body.input)],
#             'is_create_order': False
#         }, config=config, stream_mode="messages"):
#             if isinstance(chunk, AIMessageChunk):
#                 print(chunk)
#                 yield f'data:{chunk.content}\n\n'
#         yield f'data:{jsonpickle.encode({"status": "completed"})}'
#
#     return StreamingResponse(
#         generator_stream(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive",
#         }
#     )
