from langchain.agents import AgentType

from llm import llm
from langgraph.graph import StateGraph
from typing import TypedDict, Optional
from operator import add
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from constant import SQL_URL, db

# 定义工具集
tools = SQLDatabaseToolkit(
    db=db,
    llm=llm
)

sql_agent = create_sql_agent(
    llm=llm,
    toolkit=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)


