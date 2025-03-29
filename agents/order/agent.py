from langchain.agents import AgentType
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import Dict, Optional, Annotated, List, Any
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain.callbacks.base import BaseCallbackHandler
import operator
from langchain_core.callbacks import BaseCallbackManager
from constant import SQL_URL, db
from langchain_community.agent_toolkits import create_sql_agent
from sql_toolkit import SQLDatabaseToolkit
from agents.order.prompts import client_templates
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


# openai-tools
sql_agent = create_sql_agent(
    llm=llm,
    toolkit=sql_tools,
    agent_executor_kwargs={
        "return_intermediate_steps": True,
    },
    prefix="""
        You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
        You need to analyze every sentence from the user, as their first statement might involve querying customer information.  
        You should strive to search across multiple related tables as comprehensively as possible.
        You must return the primary key of each table.
        If an index exists in the table, you should use it for queries to optimize performance.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the below tools. Only use the information returned by the below tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
        
        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
        
        If the question does not seem related to the database, just return "I don't know" as the answer.
        """,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)



class OrderState(Dict):
    messages: Annotated[List[BaseMessage], operator.add]
    result: Optional[str] = None
    client: Optional[str] = None # 需要查询的客户信息
    query_client: Optional[Annotated[List[Any], operator.add]] = None # 查询到的客户信息
    query_order: Optional[Annotated[List[Any], operator.add]] = None # 查询到的库存信息

order_graph = StateGraph(OrderState)

# 尝试从用户输入中提取出客户的信息
def initial_client_node(state: OrderState):
    prompts = ChatPromptTemplate.from_messages([
        ('system', client_templates),
        ('user', """
            begin!
            {input}
        """)
    ])

    chain = prompts | llm

    result = chain.invoke({
        "input": state['messages'][-1].content
    })
    print('result', result)
    if result and result.content != '':
        return {
            'client': result.content
        }
    return state

def query_client_node(state: OrderState):
    last_message = state['messages'][-1]
    new_state = state.copy()
    new_state['query_client'] = []
    try:
        if state['client'] is not None and state['client'] != '':
            result = sql_agent.invoke({
                "input": f"""
                帮我查询客户信息,
                你可以使用 LIKE 模糊查询，你需要考虑用户可能少填写字符的情况，你需要使用模糊查询解决，示例如下: 
                `SELECT * FROM table_name WHERE name LIKE '%张%集%';`
                客户信息如下:
                {state['client']}
            """
            })
        for tool in result['intermediate_steps']:
            if tool and isinstance(tool, tuple) and tool[0] is not None and hasattr(tool[0], 'tool'):
                if tool[0].tool == 'sql_db_query' and tool[1] != '':
                    new_state['query_client'].append(jsonpickle.decode(tool[1]))
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
                    new_state['query_order'].append(jsonpickle.decode(tool[1]))
        return new_state
    except Exception as e:
        return new_state
order_graph.add_node('query_inventory_node', query_inventory_node)
order_graph.add_node('query_client_node', query_client_node)
order_graph.add_node('initial_client_node', initial_client_node)

order_graph.add_edge(START, 'initial_client_node')
order_graph.add_edge('initial_client_node', 'query_client_node')
order_graph.add_edge('query_client_node', 'query_inventory_node')

order_graph.add_edge('query_inventory_node', END)

order_agent = order_graph.compile()
