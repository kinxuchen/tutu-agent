from langgraph.graph import StateGraph, END, START
from typing import Dict, Optional, Annotated, List, Literal
from langchain_core.runnables import ConfigurableFieldSpec
import operator
from langchain_core.messages import BaseMessage
from agents.intent.agent import intent_agent
from redis.asyncio import Redis as RedisAsync
from redis import Redis
from  RedisCheckpointerSaver import  RedisCheckpointSaver


class MainAgentState(Dict):
    messages: Annotated[List[BaseMessage], operator.add] = []
    intent: Optional[Literal['chat', 'sql', 'search']] = 'chat'
    result: Optional[str] = None # 最终输出结果

main_graph = StateGraph(MainAgentState)




def start_node(state: MainAgentState):
    result = intent_agent.invoke(input={
        "messages": state['messages']
    })
    return {
        'intent': result['intent']
    }

# 判断执行哪个意图
def continue_intent_node(state: MainAgentState):
    return state['intent']

# 执行 sql 的意图
def sql_intent_node(state: MainAgentState):
    from agents.db.agent import sql_agent
    last_message = state['messages'][-1]
    result = sql_agent.invoke({
        "input": last_message.content
    })
    return {
        'result': result
    }
# 执行搜索任务
def search_intent_node(state: MainAgentState):
    pass

def chat_intent_node(state: MainAgentState):
    from llm import llm
    from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    prompt = ChatPromptTemplate.from_messages((
        [
            MessagesPlaceholder(variable_name="messages")
        ]
    ))
    chain  = RunnablePassthrough.assign(
        messages=lambda x: x['messages'],
    ) | prompt | llm
    result =  chain.invoke({
        "messages": state['messages']
    })
    return {
        "messages": [result]
    }



main_graph.add_node('start_node', start_node)
main_graph.add_node('continue_intent_node', continue_intent_node)
main_graph.add_node('sql_intent_node', sql_intent_node)
main_graph.add_node('search_intent_node', search_intent_node)
main_graph.add_node('chat_intent_node', chat_intent_node)

main_graph.add_conditional_edges('start_node', continue_intent_node, {
    'sql': 'sql_intent_node',
    'chat': 'chat_intent_node',
    'search': 'chat_intent_node'
})

main_graph.set_entry_point('start_node')

main_graph.add_edge('chat_intent_node', END)
main_graph.add_edge('sql_intent_node', END)

# 构建智能体
def create_agent(redis_async: RedisAsync, redis: Redis):
    checkpointer = RedisCheckpointSaver(redis_async, redis, 'main_agent')
    return main_graph.compile(checkpointer=checkpointer)
