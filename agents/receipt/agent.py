import jsonpickle
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from agents.receipt.tools import vector_search, clientele_vector_search
from agents.receipt.prompte import clientele_search_prompt
from components._redis import redis
from llm import llm
from typing import List, Dict, Any, Union
from agents.receipt.example_selector import few_shot_prompt
from checkpointer.RedisCheckpointerSaver import RedisCheckpointSaver
from pydash import get, every, some
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from agents.receipt.image_agent import image_receipt_agent
from agents.receipt.enum import RESUME_TYPE

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
    messages: List[BaseMessage]  # 消息列表
    resume_type: RESUME_TYPE = Field(description='中断类型', default=RESUME_TYPE.all) # 0 全部出错 1 客户信息缺失
    result: Union[List[Dict[str, Any]], None] = Field(description='最终结果', default=None)
    error_message: Union[str, None] = Field(description='错误信息', default=None)
    # 搜索重试次数
    retry: int = Field(description='重试次数', default=0)
    # 人工重试次数
    human_retry: int = Field(description='人工重试次数', default=0)
    is_small: bool = Field(description="是否是细码", default=True)


receipt_graph = StateGraph(ReceiptState)


def vector_search_node(state: ReceiptState):
    """向量工具调用"""
    messages = state.messages
    retry = state.retry
    tool_llm = llm.bind_tools(tools)
    chain = few_shot_prompt | tool_llm
    ai_message = chain.invoke({
        'input': messages[-1].content,
        'tool_names': tool_names
    })
    messages.append(ai_message)
    retry += 1
    return {
        'messages': messages,
        'retry': retry,
        'error_message': None
    }
def condition_tool_call(state: ReceiptState):
    """判断是否需要工具调用"""
    try:
        messages = state.messages
        retry = state.retry
        human_retry = state.human_retry
        last_message = messages[-1]

        if human_retry > MAX_HUMAN_RETRY:
            return 'error'
        if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
            return 'tool_call'
        elif retry < MAX_RETRY:
            return 'vector_search'
        else:
            return 'human'
    except Exception as e:
        return 'error'

def condition_tool_node(state: ReceiptState):
    """对工具节点调用后的信息进行判断"""
    messages = state.messages
    human_retry = state.human_retry
    resume_type = state.resume_type
    last_message = messages[-1]
    if human_retry > MAX_HUMAN_RETRY:
        return 'error'
    if isinstance(last_message, ToolMessage):
        try:
            tool_result = jsonpickle.decode(last_message.content)
            if resume_type == 0:
                # 全部客户都不存在的情况
                not_clientele = every(tool_result, lambda x: x['clientele_id'] is None)
                if not_clientele:
                    return 'miss_all_clientele_human_pre'
                else:
                    return 'tool_summary'
            elif resume_type == 1:
                return 'miss_all_clientele_human_pre' if tool_result is None else 'insert_clientele'
        except Exception as e:
            return 'error'
    else:
        return 'human'

def condition_image_node(state: ReceiptState):
    """判断是否是图片节点识别"""
    if hasattr(state, 'image_urls') and len(state.image_urls) > 0:
        return 'vision_images_agent'
    return 'default'
def tool_call_node(state: ReceiptState):
    """工具节点调用"""
    messages = state.messages
    last_message = messages[-1]
    for tool in last_message.tool_calls:
        tool_name = tool['name']
        if tool_name in tool_map:
            tool_fun = tool_map[tool_name]
            result = tool_fun.invoke(tool['args'])
            messages.append(
                ToolMessage(
                    content=jsonpickle.encode(result),
                    tool_call_id=tool['id']
                )
            )
    return {
        'messages': messages
    }

def tool_summary_node(state: ReceiptState):
    """
        工具调用后，处理工具函数的结果，
        此处特指向量检索
    """
    messages = state.messages
    last_message = messages[-1]
    tool_result = jsonpickle.decode(last_message.content)
    is_miss_client = some(tool_result, lambda x: x['clientele_id'] is None)
    if is_miss_client:
        first_clientele = list(filter(lambda x: x['clientele_id'] is not None, tool_result))[0]
        for x in tool_result:
            if get(x, 'clientele_id', None) is None and get(x, 'clientele', None) is None:
                x['clientele_id'] = first_clientele['clientele_id']
                x['clientele'] = first_clientele['clientele']
    return {
        'result': tool_result
    }

def miss_all_clientele_human_pre_node(state: ReceiptState):
    """
    处理缺失客户信息的情况
    """
    messages = state.messages
    last_message = messages[-1]
    result = jsonpickle.decode(last_message.content)
    return {
        'result': result,
        'resume_type': 1
    }


# 人工干预节点
def human_node(state: ReceiptState):
    # 进入人工干预阶段的，说明之前阶段的数据不可用
    human_retry = state.human_retry
    resume_type = state.resume_type
    resume_text = '客户信息缺失, 请重新输入客户信息' if state.resume_type == 1 else '无法识别用户的输入，请重新输入正确的单据信息'
    value = interrupt({
        'text': resume_text
    })
    messages = state.messages
    messages.append(HumanMessage(
        content=value
    ))
    human_retry += 1
    # 全部失败，重新向量处理
    if resume_type == 0:
        return Command(goto='vector_search_node', update={
            'messages': messages,
            'retry': 0,
            'human_retry': human_retry,
            'resume_type': 0
        })
    else:
        # 处理客户搜索
        return Command(goto='clientele_search_node', update={
            'messages': messages,
            'retry': 0,
            'human_retry': human_retry,
        })

def clientele_search_node(state: ReceiptState):
    """客户信息搜索"""
    tools = [clientele_vector_search]
    messages = state.messages
    tool_names = "-".join([f"{tool.name}\n" for tool in tools])
    tool_llm = llm.bind_tools(tools)
    chain = clientele_search_prompt | tool_llm
    ai_message = chain.invoke({
        'input': messages[-1].content,
        'tool_names': tool_names
    })
    messages.append(ai_message)
    return {
        'messages': messages
    }

# 处理客户数据插入
def insert_clientele_node(state: ReceiptState):
    messages = state.messages
    last_message = messages[-1]
    result = state.result
    clientele_dict = jsonpickle.decode(last_message.content)
    for x in result:
        x['clientele_id'] = clientele_dict['clientele_id']
        x['clientele'] = clientele_dict['clientele']
    return {
        'result': result,
        'resume_type': 0, # 恢复成默认值
    }

# 失败节点
def error_node(state: ReceiptState):
    return {
        'error_message': '智能体调用失败',
        'human_retry': 0,
        'retry': 0,
        'messages': [],
    }

receipt_graph.add_node('vector_search_node', vector_search_node)
receipt_graph.add_node('condition_tool_call', condition_tool_call)
receipt_graph.add_node('condition_image_node', condition_image_node)
receipt_graph.add_node('tool_call_node', tool_call_node)
receipt_graph.add_node('human_node', human_node)
receipt_graph.add_node('miss_all_clientele_human_pre_node', miss_all_clientele_human_pre_node)
receipt_graph.add_node('error_node', error_node)
receipt_graph.add_node('condition_tool_node', condition_tool_node)
receipt_graph.add_node('tool_summary_node', tool_summary_node)
receipt_graph.add_node('clientele_search_node', clientele_search_node)
receipt_graph.add_node('insert_clientele_node', insert_clientele_node)
receipt_graph.add_node('image_receipt_agent_node', image_receipt_agent)

receipt_graph.add_conditional_edges(START, condition_image_node, {
    'default': 'vector_search_node',
    'vision_images_agent': 'image_receipt_agent_node'
})
receipt_graph.add_conditional_edges('vector_search_node', condition_tool_call, {
    'tool_call': 'tool_call_node',
    'human': 'human_node',
    'vector_search': 'vector_search_node',
    'error': 'error_node'
})
# 工具节点调用后判断
receipt_graph.add_conditional_edges('tool_call_node', condition_tool_node, {
    'human': 'human_node', # 人工干预
    'error': 'error_node', # 失败节点
    'tool_summary': 'tool_summary_node', # 工具总结
    'insert_clientele': 'insert_clientele_node', # 插入节点
    'miss_all_clientele_human_pre': 'miss_all_clientele_human_pre_node' # 处理缺失客户信息的情况
})
receipt_graph.add_edge('miss_all_clientele_human_pre_node', 'human_node')
receipt_graph.add_edge('clientele_search_node', 'tool_call_node')
receipt_graph.add_edge('insert_clientele_node', END)
receipt_graph.add_edge('image_receipt_agent_node', END)

checkpointer = MemorySaver()
receipt_agent = receipt_graph.compile(
    # checkpointer=redisCheckpointerSaver
    checkpointer=checkpointer
)
