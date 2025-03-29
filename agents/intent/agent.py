# 意图识别，判断用户的意图是什么
from llm import llm
from langgraph.graph import StateGraph, START, END
from typing import Dict, Optional, Annotated, List, Literal
from langchain_core.messages import BaseMessage
import operator
from agents.intent.prompts import  intent_chat_prompt
from constant import db_schema

# chat 代表普通聊天任务
# sql 希望从数据库查询数据
# search 普通的搜索任务
class IntentAgentState(Dict):
    messages: Annotated[List[BaseMessage], operator.add]
    intent: Literal['chat', 'sql', 'search'] = 'chat' # 默认是一个 chat 聊天 sql 数据查询，search 配置一个 agent, order 创建订单

intent_graph = StateGraph(IntentAgentState)

# 入口节点，用于判断初始化一些数据
def start_intent_node(state: IntentAgentState):
    return 'error' if len(state['messages']) == 0 else 'intent'
# 意图识别
def intent_node(state: IntentAgentState):
    last_message = state['messages'][-1]
    function_calling_schema = {
        "name": "IntentClassifier",  # 函数名
        "description": "对 LLM 结果进行约束化处理", # 函数描述
        "parameters": {          # 函数参数的 JSON Schema
            "type": "object",
            "properties": {
                "intent": {      # 参数名 (你最终想要提取的值)
                    "type": "string",
                    "description": "The classified intent", # 参数描述 (可选)
                    "enum": ['sql', 'chat', 'search'] # 参数的约束
                }
            },
            "required": ["intent"] # 指定 'intent' 参数是必需的
        }
    }

    schema_llm = llm.with_structured_output(
        schema=function_calling_schema,
        method='function_calling'
    )
    chain = intent_chat_prompt | schema_llm
    result = chain.invoke(input={
        'input': last_message.content,
        'schema': db_schema
    })
    if isinstance(result, dict) and 'intent' in result:
        return result
    return {
        'intent': 'chat'
    }

def error_handle_node(state: IntentAgentState):
    return {
        'intent': 'chat'
    }

intent_graph.add_node('start_intent_node', start_intent_node)
intent_graph.add_node('intent_node', intent_node)
intent_graph.add_node('error_handle_node', error_handle_node)

intent_graph.add_conditional_edges(START, start_intent_node, {
    'intent': 'intent_node',
    'error': 'error_handle_node'
})

intent_graph.add_edge('error_handle_node', END)
intent_graph.add_edge('intent_node', END)


intent_agent = intent_graph.compile()
