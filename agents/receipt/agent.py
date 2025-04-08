from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from components.store import get_vector_store, milvus_vector_store
from llm import llm, embeddings
from constant import COLLECTION_TUTU_NAME, PARTITION_STORE_NAME, PARTITION_CLIENTELE_NAME
from typing import Annotated, List, Dict, Any
from agents.receipt.example_selector import few_shot_prompt
from pydash import get


# 维护基准的客户信息
class GoodResult(BaseModel):
    name: str = Field(description='商品名称,解析不出可以为空')
    color: str = Field(description='颜色,解析不出可以为空')
    count: int = Field(description='数量,解析不出可以为空')
    clientele: str = Field(description='客户信息,解析不出可以为空')


class GoodsResults(BaseModel):
    goods: List[GoodResult] = Field(description='商品列表')


class ReceiptState(BaseModel):
    messages: List[BaseMessage]  # 消息列表
    result: List[Dict[str, Any]] = Field(description='最终结果', default={})
    goods: GoodsResults = Field(description="匹配到的商品", default=None)  # 匹配到的商品
    error_message: str = Field(description='错误信息', default=None)


receipt_graph = StateGraph(ReceiptState)


def analyze_receipt_node(state: ReceiptState):
    """解析用户的输入，提取出客户信息和库存信息的结构"""
    last_message = state.messages[-1]
    structured_llm = llm.with_structured_output(
        schema=GoodsResults,
        method='function_calling'
    )
    chain = few_shot_prompt | structured_llm
    goods = chain.invoke({
        'input': last_message.content,
    })
    return {
        'goods': goods
    }


def condition_goods_node(state: ReceiptState):
    """
    根据提示词解析出的结果，去向量数据库中查询, 如果没有解析出相关的信息。则返回空
    """
    return 'vector_search_node' if state.goods is not None else 'error_node'


def vector_search_node(state: ReceiptState):
    """
    从 LLM 结构化用户输入后，从 zilliz cloud 中获取用户输入
    :param state:
    :return:
    """
    vector_store = get_vector_store()
    # 搜索结果，先过滤是否存在商品名称，客户信息可以是空的。
    try:
        search_data = [
            {
                'sku_result': vector_store.search(
                    data=[embeddings.embed_query(
                        f"""商品名称: {good.name}\n商品颜色: {good.color}"""
                    )],
                    collection_name=COLLECTION_TUTU_NAME,
                    anns_field='vector',  # 需要查询的字段
                    partition_names=[PARTITION_STORE_NAME],
                    limit=1,
                    output_fields=['id', 'db_id', 'metadata'],
                    search_params={
                        "params": {
                            'radius': 0.7,
                            'level': 5,
                            'range_filter': 1,
                            'enable_recall_calculation': True
                        }
                    }
                ),
                'clientele_result': [] if (good.clientele is None or good.clientele == '') else vector_store.search(
                    data=[embeddings.embed_query(
                        f"""{good.clientele}"""
                    )],
                    collection_name=COLLECTION_TUTU_NAME,
                    anns_field='vector',  # 需要查询的字段
                    partition_names=[PARTITION_CLIENTELE_NAME],
                    limit=1,
                    output_fields=['id', 'db_id', 'metadata'],
                    search_params={
                        "params": {
                            'radius': 0.7,
                            'level': 5,
                            'enable_recall_calculation': True
                        }
                    }
                ),
                'name': good.name,
                'color': good.color,
                'clientele': good.clientele,
            } for good in filter(lambda x: x.name is not None, state.goods.goods)
        ]

        return {
            'result': list(
                map(lambda x: {
                    'name': None if len(x['sku_result']) == 0 else get(x, ['sku_result', 0, 0, 'entity', 'metadata', 'name'], default=None),
                    'good_id': None if len(x['sku_result']) == 0 else get(x, ['sku_result', 0, 0, 'entity', 'metadata', 'id'], default=None),
                    'color': None if len(x['sku_result']) == 0 else get(x, ['sku_result', 0, 0, 'entity', 'metadata', 'color'], default=None),
                    'clientele': None if len(x['clientele_result']) == 0 else get(x, ['clientele_result', 0, 0, 'entity', 'metadata', 'name'], default=None),
                    'clientele_id': None if len(x['clientele_result']) == 0 else get(x, ['clientele_result', 0, 0, 'entity', 'metadata', 'id'], default=None),
                }, search_data)
            )
        }
    except Exception as e:
        return {
            'error_message': str(e)
        }

def condition_base_info_node(state: ReceiptState):
    """
    判断是否从向量数据库中获取数据
    :param state: ReceiptState
    :return: state ReceiptState
    """
    pass
# 执行错误的节点
def error_node(state: ReceiptState):
    return {
        'error_message': '没有匹配到库存商品信息'
    }



receipt_graph.add_node('init_receipt_node', analyze_receipt_node)
receipt_graph.add_node('continue_goods_node', condition_goods_node)
receipt_graph.add_node('vector_search_node', vector_search_node)
receipt_graph.add_node('error_node', error_node)

receipt_graph.add_edge(START, 'init_receipt_node')
receipt_graph.add_conditional_edges('init_receipt_node', condition_goods_node, {
    'vector_search_node': 'vector_search_node',
    'error_node': 'error_node'
})
receipt_graph.add_edge('init_receipt_node', END)
receipt_graph.add_edge('error_node', END)
receipt_graph.add_edge('vector_search_node', END)

receipt_agent = receipt_graph.compile()
