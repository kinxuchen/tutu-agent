from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from agents.rag.tool import knowledge_search_tool
from llm import llm
from functools import reduce
from agents.rag.prompts import knowledge_search_prompts
import jsonpickle
from components._redis import redis
from checkpointer.RedisCheckpointerSaver import  RedisCheckpointSaver

redisCheckpointerSaver = RedisCheckpointSaver(
    redis_client=redis,
    prefix='rag'
)

# 定义需要使用的工具
tools = [knowledge_search_tool]
tool_names = "\n".join(f"- {tool.name}" for tool in tools)
tool_map = {
    tool.name: tool for tool in tools
}

class RagState(BaseModel):
    input: str = Field(description='当前的用户输入', default=None)
    messages: List[BaseMessage] = Field(description='消息列表', default=[])
    urls: List[str] = Field(description='知识库的url', default=[])

rag_graph = StateGraph(RagState)

def knowledge_search_node(state: RagState):
    """调用大模型，从知识库中查询数据"""
    messages = state.messages
    tool_llm = llm.bind_tools(tools)
    prompts = knowledge_search_prompts.partial(tool_names=tool_names)
    chain = prompts | tool_llm
    ai_message = chain.invoke(input={
        'input': state.input
    })
    # 将 AI 结果插入消息列表中
    messages.append(ai_message)
    return {
        'messages': messages
    }

def condition_tool_node(state: RagState):
    """对工具节点调用后的信息进行判断"""
    messages = state.messages
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
        return 'tool_call_node'
    else:
        return END

def tool_call_node(state: RagState):
    """对工具调用结果进行调用"""
    messages = state.messages
    last_message = messages[-1]
    for tool in last_message.tool_calls:
        tool_name = tool['name']
        if tool_name in tool_map:
            tool_fun = tool_map[tool_name]
            result = tool_fun.invoke(tool['args'])
            if tool['name'] == 'knowledge_search_tool' and len(result) > 0:
                tool_message = ToolMessage(
                    content="\n\n".join(f"- {item.page_content}" for item in result),
                    tool_call_id=tool['id'],
                    tool_name=tool['name'],
                    additional_kwargs={
                        "urls": list(map(lambda x: x.metadata['url'], result))
                    }
                )
                messages.append(tool_message)
            # 工具调用结果

    return {
        'messages': messages
    }

def summary_node(state: RagState):
    """根据工具调用结果和用户消息，再次发送"""
    messages = state.messages
    # 工具消息
    last_message = messages[-1]
    # 构建一条用户消息
    messages.append(HumanMessage(
        content=f"""
        请你根据工具函数的调用结果，解答我下面的问题
        {state.input}
        """
    ))
    result = llm.invoke(messages)
    print('result', result)
    messages.append(result)
    return {
        'messages': messages,
        'urls': last_message.additional_kwargs['urls']
    }
rag_graph.add_node('knowledge_search_node', knowledge_search_node)
rag_graph.add_node('condition_tool_node', condition_tool_node)
rag_graph.add_node('tool_call_node', tool_call_node)
rag_graph.add_node('summary_node', summary_node)

rag_graph.add_edge(START, 'knowledge_search_node')
rag_graph.add_conditional_edges('knowledge_search_node', condition_tool_node, {
    'tool_call_node': 'tool_call_node'
})
rag_graph.add_edge('tool_call_node', 'summary_node')
rag_graph.add_edge('summary_node', END)

rag_agent = rag_graph.compile(checkpointer=redisCheckpointerSaver)
