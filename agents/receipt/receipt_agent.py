from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START,END
from agents.receipt.example_selector import few_shot_prompt
from llm import llm
from agents.receipt.enum import (
    RESUME_TYPE,
    TASK_TYPE,
)
from agents.receipt.tools import vector_search


class State(BaseModel):
    input: str = Field(description="当前用户输入", default=None)
    drafts: List[str] = Field(description="用户输入草稿", default=[]) # 管理用户输入草稿
    messages: List[BaseMessage] = Field(description="消息列表", default=[])  # 消息列表
    resume_type: RESUME_TYPE = Field(description='中断类型', default=RESUME_TYPE.all) # 0 全部出错 1 客户信息缺失
    task_type: TASK_TYPE = Field(description='任务类型', default=TASK_TYPE.sell)
    result: Union[List[Dict[str, Any]], None] = Field(description='最终结果', default=None)
    error_message: Union[str, None] = Field(description='错误信息', default=None)
    reply_message: Union[str, None] = Field(description='回复信息', default=None)
    # 搜索重试次数
    retry: int = Field(description='重试次数', default=0)
    # 人工重试次数
    human_retry: int = Field(description='人工重试次数', default=0)
    # 是否是细码
    is_small: bool = Field(description="是否是细码", default=True)

receipt_graph = StateGraph(State)
# 检查是否执行草稿节点
def condition_draft_node(state: State):
    if state.input == '确定':
        # 判断当前是否存在草稿，有草稿则指定单据创建，没有草稿则执行回复
        return 'apply_draft' if len(state.drafts) > 0 else 'apply_reply' # 执行草稿创建单据操作
    return 'apply_reply' # 执行回复

# 继续输入
def continue_input_node(state: State):
    drafts = state.drafts
    # 将当前用户消息存入草稿列表
    drafts.append(state.input)
    value = interrupt({
        'message': '是否继续输入单据信息，如果已经完成，请回复【确定】' if len(drafts) > 0 else '还有输入单据信息，请先输入单据信息，输入完成后可以回复【确认】'
    })
    # 重新执行到判断是否执行草稿节点阶段
    return Command(
        goto="condition_draft_node",
        update={
            'input': value,
            'drafts': drafts,
        }
    )

# 执行草稿阶段
def apply_draft_node(state: State):
    human_message = HumanMessage(
        content="\n".join([f"- {content}" for content in state.drafts])
    )
    tools = llm.bind_tools()
    chain = few_shot_prompt
    pass

receipt_graph.add_node('continue_input_node', continue_input_node)
receipt_graph.add_node('condition_draft_node', condition_draft_node)
receipt_graph.add_conditional_edges(START, condition_draft_node, {
    'apply_draft': END,
    'apply_reply': 'continue_input_node',
})


receipt_node = receipt_graph.compile()
