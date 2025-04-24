from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from agents.receipt.tools import vector_search, clientele_vector_search
from components._redis import redis
from typing import List, Dict, Any, Union
from checkpointer.RedisCheckpointerSaver import RedisCheckpointSaver
from agents.receipt.image_agent import image_receipt_agent
from agents.receipt.receipt_agent import receipt_node
from agents.receipt.enum import (
    TASK_TYPE,
)

MAX_RETRY = 2
MAX_HUMAN_RETRY = 1

tools = [vector_search]
tool_names = "- ".join([f"{tool.name}\n" for tool in tools])
tool_map = {tool.name: tool for tool in tools}
tool_map['clientele_vector_search'] = clientele_vector_search

redisCheckpointerSaver = RedisCheckpointSaver(redis, 'main_agent')

# 维护基准的客户信息

class ReceiptState(BaseModel):
    image_urls: List[str] = Field(description='图片地址', default=None)
    input: str = Field(description="当前用户输入", default=None)
    # messages: List[BaseMessage] = Field(description="消息列表", default=[])  # 消息列表
    task_type: TASK_TYPE = Field(description='任务类型', default=TASK_TYPE.sell)
    result: Union[List[Dict[str, Any]], None] = Field(description='最终结果', default=None)
    error_message: Union[str, None] = Field(description='错误信息', default=None)
    # 是否是细码
    is_small: bool = Field(description="是否是细码", default=True)

receipt_graph = StateGraph(ReceiptState)

# 判断是否执行图片节点
def condition_images_node(state: ReceiptState):
    if hasattr(state,'image_urls') and len(state.image_urls) > 0:
        return 'image_node'
    return 'default'

# 新增一个转换节点，用于转换主图和子图的状态
def conversion_receipt_node(state: ReceiptState):
    return {
        'is_small': state.is_small,
        'task_type': state.task_type,
        'input': state.input
    }

receipt_graph.add_node('condition_images_node', condition_images_node)
receipt_graph.add_node('image_node', image_receipt_agent)
receipt_graph.add_node('receipt_node', receipt_node)
receipt_graph.add_node('conversion_receipt_node', conversion_receipt_node)

receipt_graph.add_conditional_edges(START, condition_images_node, {
    'image_node': 'image_node',
    'default': 'conversion_receipt_node'
})

receipt_graph.add_edge('conversion_receipt_node', 'receipt_node')

