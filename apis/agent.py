from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from dto.chat_request_dto import ReceiptRequestBody
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from agents.receipt.agent import receipt_agent, ReceiptState
from pydash import has, get

agent_router = APIRouter(prefix="/agent")
# 单据接口
@agent_router.post('/receipt/{user_id}/{thread_id}')
async def receipt_agent_request(user_id:str, thread_id: str, body: ReceiptRequestBody):
    """
    根据用户的输入生成单据信息
    Args:
        user_id: 用户id
        thread_id: 会话id
        body: 单据请求体
    Returns:
        success: 是否成功
    """
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