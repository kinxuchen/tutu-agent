from fastapi import APIRouter, HTTPException
from agents.agent import Agent
from dto.chat_request_dto import ReceiptRequestBody
from langchain_core.messages import HumanMessage
from langgraph.types import Command
import traceback
from agents.receipt.agent import receipt_agent, ReceiptState
from agents.test_agent import graph as test_agent
from pydash import has, get

agent_router = APIRouter(prefix="/agent")

agent_instance = Agent()
@agent_router.post('/chat/{user_id}/{thread_id}')
async def chat(user_id: str, thread_id: str, body: ReceiptRequestBody):
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
async def receipt_agent_request(user_id:str, thread_id: str, body: ReceiptRequestBody):
    config = {
        'configurable': {
            'thread_id': thread_id,
            "user_id": user_id
        }
    }
    agent_state = receipt_agent.get_state(config=config)
    is_resume = has(agent_state, ['tasks', 0])
    agent_result = None
    # 中断恢复执行
    if is_resume:
        agent_result = receipt_agent.invoke(
            input=Command(
                resume=body.input
            ),
            config=config
        )
    else:
        # 第一次输入，需要初始化状态
        agent_result = receipt_agent.invoke(input={
            'messages': [HumanMessage(content=body.input)],
            'result': None,
            'resume_type': 0,
            'error_message': None,
            'retry': 0,
            'is_small': body.is_small,
            'human_retry': 0,
            'image_urls': body.image_urls
        }, config=config)

    state = receipt_agent.get_state(config=config)
    is_recover = has(state, ['tasks', 0])
    # 执行完成判断是否是中断
    if is_recover:
        resume_text = get(state, ['tasks', 0, 'interrupts', 0, 'value', 'text'])
        return {
            'success': True,
            'is_resuming': True,
            'message': resume_text
        }
    else:
        if agent_result.get('error_message'):
            return {
                'success': False,
                'message': agent_result.get('error_message')
            }
        return {
            'success': True,
            'data': agent_result.get('result', None)
        }

@agent_router.post('/test_agent/{thread_id}')
async def test_agent_request(thread_id: str, body: ReceiptRequestBody):
    config = {
        'configurable': {
            'thread_id': thread_id
        }
    }
    result = test_agent.invoke(input={
        'some_text': body.input
    }, config=config)

    return {
        'success': True
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
