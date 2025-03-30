from langchain.agents import AgentType
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import Dict, Optional, Annotated, List, Any, override
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain.callbacks.base import BaseCallbackHandler
import operator
from constant import SQL_URL, db
from langchain_community.agent_toolkits import create_sql_agent
from sql_toolkit import SQLDatabaseToolkit
from agents.order.prompts import clientele_templates, SQL_PREFIX
from llm import llm
import jsonpickle
from uuid import UUID


# 获取数据库上下文
db_context = db.get_context()
@tool
def create_order():
    """你的职责是提取用户数据库中的相应字段"""
    pass

sql_tools = SQLDatabaseToolkit(
    db=db,
    llm=llm
)


class ToolCallbackHandle(BaseCallbackHandler):
    @override
    def on_tool_end(
            self,
            output: Any,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ):
        print('output', output)

tool_callback = ToolCallbackHandle()
# openai-tools
sql_agent = create_sql_agent(
    llm=llm,
    toolkit=sql_tools,
    agent_executor_kwargs={
        "return_intermediate_steps": True,
    },
    callbacks=[tool_callback],
    prefix=SQL_PREFIX,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)



class OrderState(Dict):
    messages: Annotated[List[BaseMessage], operator.add]
    result: Optional[str] = None
    clientele: Optional[str] = None # 需要查询的客户信息
    query_client: Optional[Annotated[List[Any], operator.add]] = None # 查询到的客户信息
    query_order: Optional[Annotated[List[Any], operator.add]] = None # 查询到的库存信息

order_graph = StateGraph(OrderState)

# 尝试从用户输入中提取出客户的信息
def initial_clientele_node(state: OrderState):
    prompts = ChatPromptTemplate.from_messages([
        ('system', clientele_templates),
        ('user', """
            begin!
            {input}
        """)
    ])

    json_schema = {
        "title": "CustomerInformationExtractor",  # 添加一个描述性的标题
        "description": "提取客户的详细信息", # 添加一个描述
        "type": "object",
        "properties": {
            "clientele": {
                "type": "string",
                "description": "客户信息" # "Customer information"
            }
        },
        "required": ["clientele"] # 最好也明确指出哪些字段是必需的
    }
    chain = prompts | llm.with_structured_output(json_schema, method="function_calling")

    result = chain.invoke({
        "input": state['messages'][-1].content
    })
    print('result', result)
    if result['clientele'] is not None and result['clientele'] != '':
        return result
    return state

def query_clientele_node(state: OrderState):
    last_message = state['messages'][-1]
    new_state = state.copy()
    new_state['query_client'] = []
    try:
        if state['clientele'] is not None and state['clientele'] != '':
            result = sql_agent.invoke({
                "input": f"""
                Help me check customer information
                You can use the LIKE fuzzy query. You need to account for cases where users might input incomplete characters, and you should resolve this by using fuzzy queries. Here's an example:
                `SELECT * FROM table_name WHERE name LIKE '%张%集%';`
                The customer information is as follows:
                {state['clientele']}
            """
            })
        for tool in result['intermediate_steps']:
            if tool and isinstance(tool, tuple) and tool[0] is not None and hasattr(tool[0], 'tool'):
                if tool[0].tool == 'sql_db_query' and tool[1] != '':
                    new_state['query_client'] += jsonpickle.decode(tool[1])
        return new_state
    except Exception as e:
        return new_state
# 1. 查询库存数据
def query_inventory_node(state: OrderState):
    last_message = state['messages'][-1]
    new_state = state.copy()
    new_state['query_order'] = []
    try:
        if last_message:
            result = sql_agent.invoke({
                "input": last_message.content
            })
        for tool in result['intermediate_steps']:
            if tool and isinstance(tool, tuple) and tool[0] is not None and hasattr(tool[0], 'tool'):
                if tool[0].tool == 'sql_db_query' and isinstance(tool[1], str) and tool[1] != '':
                    new_state['query_order'] += jsonpickle.decode(tool[1])
        return new_state
    except Exception as e:
        return new_state



order_graph.add_node('query_inventory_node', query_inventory_node)
order_graph.add_node('query_clientele_node', query_clientele_node)
order_graph.add_node('initial_clientele_node', initial_clientele_node)

order_graph.add_edge(START, 'initial_clientele_node')
order_graph.add_edge('initial_clientele_node', 'query_clientele_node')
order_graph.add_edge('query_clientele_node', 'query_inventory_node')

order_graph.add_edge('query_inventory_node', END)

order_agent = order_graph.compile()
