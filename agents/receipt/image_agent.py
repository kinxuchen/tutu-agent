"""处理图片识别相关的 Agent """
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List, Any, Union,Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from llm import llm, doubao_img_llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from agents.receipt.receipt_dto import SmallGoodsResults,ThickGoodsResults
from langchain_core.runnables import RunnableParallel, RunnableLambda
from agents.receipt.prompte import small_system_template, thick_system_template
from uuid import uuid4
from pydash import get
from constant import (
    COLLECTION_TUTU_NAME,
    PARTITION_STORE_NAME,
    PARTITION_CLIENTELE_NAME
)
from utils.vector import vector_search

MAX_RETRY = 2 # 重试次数

class ImageReceiptState(BaseModel):
    image_urls: List[str] = Field(description='图片地址', default=None)
    messages: List[BaseMessage] = Field(description='消息列表', default=[])
    image_recognize: Any = Field(description='图片识别结果', default=None)
    vector_result: Union[Dict[str, Any], None] = Field(description='向量搜索结果', default=None)
    result: Any = Field(description='最终结果', default=None)
    is_small: bool = Field(description="是否是细码", default=True)
    retry_count: int = Field(description='重试次数', default=0)


image_receipt_graph = StateGraph(ImageReceiptState)

def vision_images_node(state: ImageReceiptState):
    """图片识别节点"""
    messages = state.messages
    last_message = messages[-1]
    parser = PydanticOutputParser(pydantic_object=SmallGoodsResults if state.is_small else ThickGoodsResults)
    images = [image for image in map(lambda x: {
        'type': 'image_url',
        'image_url': {
            'url': x
        }
    }, state.image_urls)]
    # 后续输出需要新增对格式判断
    fixing_parser = OutputFixingParser.from_llm(
        llm=llm,
        parser=parser,
        max_retries=2
    )
    chat_messages = ChatPromptTemplate.from_messages([
        ('system', small_system_template if state.is_small else thick_system_template),
        MessagesPlaceholder(variable_name="messages"),
    ]).partial(format_instructions=parser.get_format_instructions())
    human_message = HumanMessage(
        content=[{
            "type": "text",
            'text': f"""
                {last_message.content}
            """,
        }] + images
    )
    # 删除最后一条消息，替换成组装成带图的图片
    messages.pop(-1)
    messages.append(human_message)
    chain = chat_messages | doubao_img_llm | RunnableParallel(
        json_result=fixing_parser,
        source=RunnableLambda(lambda x: x)
    )
    try:
        image_result = chain.invoke({
            'messages': messages
        })
        ai_message = AIMessage(
            content=image_result['json_result'].model_dump_json(),
            id=image_result['source'].id
        )
        messages.append(ai_message)
        return {
            'messages': messages,
            'image_recognize': image_result['json_result']
        }
    except Exception as e:
        print(e)
        return {
            'messages': messages,
            'image_recognize': None
        }

# 结果判断节点
def condition_result_node(state: ImageReceiptState):
    retry_count = state.retry_count
    result = state.image_recognize
    # 如果重试次数超过最大限制，则跳转到结束
    if retry_count > MAX_RETRY:
        return 'error'
    if result is None:
        return 'retry_vision'
    return 'vector_search' # 根据图片结果去向量数据库中查询


# 失败节点
def error_node(state: ImageReceiptState):
    return {
        'result': None,
        'retry_count': 0 # 失败后重试次数重置为 0
    }

# 重试视觉识别
def retry_vision_node(state: ImageReceiptState):
    return {
        'retry_count': state.retry_count + 1
    }

# 数据库查询节点
def vector_search_node(state: ImageReceiptState):
    try:
        # 转换成 dict
        image_recognize_list = state.image_recognize.model_dump().get('goods', [])
        for goods in image_recognize_list:
            # 为每一条任务创建一个唯一 id
            goods['task_id'] = str(uuid4())
            # 向量检索库存数据
            vector_goods_name = vector_search(
                query=f"商品名称:{goods['name']};商品颜色:{goods['color']}",
                collection_name=COLLECTION_TUTU_NAME,
                partition_names=[PARTITION_STORE_NAME],
                output_fields=['id', 'db_id', 'metadata']
            )
            # 向量检索客户信息
            vector_clientele = vector_search(
                query=goods.get('clientele'),
                collection_name=COLLECTION_TUTU_NAME,
                partition_names=[PARTITION_CLIENTELE_NAME],
                output_fields=['id', 'db_id', 'metadata']
            )
            if vector_clientele is not None:
                goods['clientele'] = get(vector_clientele, [0,0,'entity', 'metadata', 'name'], None)
                goods['clientele_id'] = get(vector_clientele, [0, 0, 'entity', 'metadata', 'id'], None)
            if vector_goods_name is not None:
                goods['name'] = get(vector_goods_name, [0,0,'entity','metadata', 'name'], None)
                goods['color'] = get(vector_goods_name, [0,0,'entity','metadata', 'color'], None)
                goods['good_id'] = get(vector_goods_name, [0,0,'entity','metadata', 'id'], None)
        return {
            'vector_result': image_recognize_list,
            'result': {
                'vector_result': image_recognize_list,
                'image_recognize': state.image_recognize
            }
        }
    except Exception as e:
        return {
            'messages': state.messages,
            'result': None
        }


image_receipt_graph.add_node('vision_images_node', vision_images_node) # 视觉识别节点
image_receipt_graph.add_node('retry_vision_node', retry_vision_node) # 重新视觉识别节点
image_receipt_graph.add_node('error_node', error_node) # 失败节点
image_receipt_graph.add_node('vector_search_node', vector_search_node) # 向量数据库查询节点


image_receipt_graph.add_edge(START, 'vision_images_node')
image_receipt_graph.add_conditional_edges('vision_images_node', condition_result_node, {
    'error': 'error_node', # 失败的节点
    'retry_vision': 'retry_vision_node', # 重新视觉识别节点
    'vector_search': 'vector_search_node' # 向量数据库查询节点
})
image_receipt_graph.add_edge('retry_vision_node', 'vision_images_node')
image_receipt_graph.add_edge('error_node', END) # 失败节点直接调转结束
image_receipt_graph.add_edge('vision_images_node', 'vector_search_node') #
image_receipt_graph.add_edge('vector_search_node', END)
image_receipt_agent = image_receipt_graph.compile()
