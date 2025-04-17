from langgraph.graph import StateGraph, END
from typing import Dict, Optional, Annotated, List, Literal
from langchain_core.messages import BaseMessage
from agents.intent.agent import intent_agent
from agents.order.agent import order_agent
from redis.asyncio import Redis as RedisAsync
from redis import Redis
from agents.prompt import summary_template
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from checkpointer.RedisCheckpointerSaver import  RedisCheckpointSaver

MAX_SIZE = 5

class MainAgentState(Dict):
    messages: List[BaseMessage] = []
    is_create_order: Optional[bool] = False
    intent: Optional[Literal['chat', 'sql', 'search', 'order']] = 'chat'
    result: Optional[str] = None # 最终输出结果

main_graph = StateGraph(MainAgentState)


def start_node(state: MainAgentState):
    if state['is_create_order']:
        return {
            'intent': 'order'
        }
    else:
        result = intent_agent.invoke(input={
            "messages": state['messages']
        })
        return {
            'intent': result['intent']
        }

# 总结摘要/切片 可以在这个步骤加一个 summary 总结的功能
def summary_messages_node(state:MainAgentState):
    from llm import llm
    messages = state['messages']
    if len(messages) > MAX_SIZE:
        new_messages = messages[0:-1]
        chain = ChatPromptTemplate.from_messages([
            ('system', summary_template),
            MessagesPlaceholder(variable_name="messages")
        ]) | llm
        summary_result = chain.invoke({
            "messages": new_messages
        })
        print('总结结果', summary_result)
        return {
            "messages": [summary_result, messages[-1]]
        }
    return state



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

# 定义会话聊天
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
        "messages": state['messages'] + [result]
    }

def order_intent_node(state: MainAgentState):
    result = order_agent.invoke({
        "messages": state['messages']
    })
    return {
        'result': result
    }



main_graph.add_node('start_node', start_node)
main_graph.add_node('summary_messages_node', summary_messages_node)
main_graph.add_node('continue_intent_node', continue_intent_node)
main_graph.add_node('sql_intent_node', sql_intent_node)
main_graph.add_node('search_intent_node', search_intent_node)
main_graph.add_node('chat_intent_node', chat_intent_node)
main_graph.add_node('order_intent_node', order_intent_node)


main_graph.add_edge('summary_messages_node', 'start_node')
main_graph.add_conditional_edges('start_node', continue_intent_node, {
    'sql': 'sql_intent_node',
    'chat': 'chat_intent_node',
    'search': 'chat_intent_node',
    'order': 'order_intent_node'
})

main_graph.set_entry_point('summary_messages_node')

main_graph.add_edge('chat_intent_node', END)
main_graph.add_edge('sql_intent_node', END)
main_graph.add_edge('order_intent_node', END)

# 构建智能体
def create_agent(redis: Redis):
    checkpointer = RedisCheckpointSaver(redis, 'main_agent')
    agent = main_graph.compile(checkpointer=checkpointer)
    return agent

