import jsonpickle
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from sqlalchemy import exc
from agents.receipt.tools import vecotr_search, clientele_vector_search
from agents.receipt.receipt_dto import GoodsResults
from agents.receipt.prompte import clientele_search_prompt
from llm import llm
from typing import List, Dict, Any
from constant import COLLECTION_TUTU_NAME, PARTITION_CLIENTELE_NAME
from components.store import get_vector_store
from agents.receipt.example_selector import few_shot_prompt
from checkpointer.RedisCheckpointerSaver import RedisCheckpointSaver
from constant import REDIS_DB, REDIS_PORT, REDIS_HOST
from redis.asyncio import Redis as RedisAsync
from redis import Redis
from pydash import get, every, some
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

MAX_RETRY = 2
MAX_HUMAN_RETRY = 1

tools = [vecotr_search]
tool_names = "- ".join([f"{tool.name}\n" for tool in tools])
tool_map = {tool.name: tool for tool in tools}
tool_map['clientele_vector_search'] = clientele_vector_search


redis_async = RedisAsync(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
redis = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

redisCheckpointerSaver = RedisCheckpointSaver(redis_async, redis, 'main_agent')

# 维护基准的客户信息

class ReceiptState(BaseModel):
    messages: List[BaseMessage]  # 消息列表
    resume_type: int = Field(description='中断类型', default=0) # 0 全部出错 1 客户信息缺失
    result: List[Dict[str, Any]] = Field(description='最终结果', default={})
    error_message: str = Field(description='错误信息', default=None)
    # 搜索重试次数
    retry: int = Field(description='重试次数', default=0)
    human_retry: int = Field(description='人工重试次数', default=0)


receipt_graph = StateGraph(ReceiptState)


# 处理数据
def vector_search_node(state: ReceiptState):
    # todo 将所有人的语言组合起来去输入
    messages = state.messages
    retry = state.retry
    print(tool_names)
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
        'retry': retry
    }

def condintion_tool_call(state: ReceiptState):
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

# 对工具节点调用后的信息进行判断
def condition_tool_node(state: ReceiptState):
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

# 工具调用
def tool_call_node(state: ReceiptState):
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



# 工具总结
def tool_summary_node(state: ReceiptState):
    messages = state.messages
    last_message = messages[-1]
    tool_result = jsonpickle.decode(last_message.content)
    is_miss_client = some(tool_result, lambda x: x['clientele_id'] is None)
    if is_miss_client:
        clientele_id = None
        clientele = None
        first_clientele = list(filter(lambda x: x['clientele_id'] is not None, tool_result))[0]
        for x in tool_result:
            if not x.clientele_id and not x.clientele:
                clientele_id = first_clientele['clientele_id']
                clientele = x['clientele']
    return {
        'result': tool_result
    }

# 处理缺失客户信息的情况
def miss_all_clientele_human_pre_node(state: ReceiptState):
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
    """处理客户数据搜索"""
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
    
    pass

    
# 失败节点
def error_node(state: ReceiptState):
    return {
        'error_message': '智能体调用失败'
    }

receipt_graph.add_node('vector_search_node', vector_search_node)
receipt_graph.add_node('condintion_tool_call', condintion_tool_call)
receipt_graph.add_node('tool_call_node', tool_call_node)
receipt_graph.add_node('human_node', human_node)
receipt_graph.add_node('miss_all_clientele_human_pre_node', miss_all_clientele_human_pre_node)
receipt_graph.add_node('error_node', error_node)
receipt_graph.add_node('condition_tool_node', condition_tool_node)
receipt_graph.add_node('tool_summary_node', tool_summary_node)
receipt_graph.add_node('clientele_search_node', clientele_search_node)
receipt_graph.add_node('insert_clientele_node', insert_clientele_node)

receipt_graph.add_edge(START, 'vector_search_node')
receipt_graph.add_conditional_edges('vector_search_node', condintion_tool_call, {
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

checkpointer = MemorySaver()
receipt_agent = receipt_graph.compile(
    # checkpointer=redisCheckpointerSaver
    checkpointer=checkpointer
)
