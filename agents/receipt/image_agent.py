"""处理图片识别相关的 Agent """
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List, Any
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from llm import image_llm, llm, doubao_llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import RetryOutputParser, OutputFixingParser
from agents.receipt.receipt_dto import GoodsResults
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda


class ImageReceiptState(BaseModel):
    image_urls: List[str] = Field(description='图片地址', default=None)
    messages: List[BaseMessage] = Field(description='消息列表', default=[])
    result: Any = Field(description='生成单据的结果', default=None)
image_receipt_graph = StateGraph(ImageReceiptState)

def vision_images_node(state: ImageReceiptState):
    messages = state.messages
    last_message = messages[-1]
    parser = PydanticOutputParser(pydantic_object=GoodsResults)
    images = [image for image in map(lambda x: {
        'type': 'image_url',
        'image_url': {
            'url': x
        }
    }, state.image_urls)]
    # 后续输出需要新增对格式判断
    system_template = """
          ## 角色
            你是一名具有丰富经验的商人，你平时和客户交易需要经常使用单据记录一些基本信息。
          ## 任务
            你现在需要尝试将一些图片中的单据信息进行提取，找到关键的信息，比如：
            - 商品名称
            - 商品颜色
            - 商品件数 (数量)
            - 商品米数 (长度)
            - 客户信息 （不是发货人）
            - 交易金额
          ## 注意
            - 商品的名称中可能包含颜色信息，你需要将颜色信息从商品信息中拆分出来，例如：
                - 商品名：黑色大金貂
                - 输出：商品:大金貂,颜色:黑色
                - 商品名: 本白大金貂喷黄花卉
                - 输出: 商品: 大金貂喷黄花卉,颜色: 本白
            - 如果图片是一个表格，在解析商品信息时，你需要忽略表格整体统计的那一行数据，只关心商品对应那一行的数据
            - 如果是表格，你需要找到表格中你解析的商品项对应那一行数据的合计(统计)数据作为总米数。一定要是对应那一行的数据，不要提取非商品行的数据
            - 商品可能存在两种单位，一种是件，一种是米，一件可能对应若干米。如果能够识别，请你尽量提取两种单位，如果没有，以件为基准单位
            - 商品的颜色(色别)和名称(品名)组成一个唯一的 SKU，如果一个相同的商品名称颜色不同，请作为两件商品处理
            - 输出内容不要有多余内容，请严格按照【输出格式】输出内容
        ## 输出格式
        {format_instructions}
    """
    fixing_parser = OutputFixingParser.from_llm(
        llm=llm,
        parser=parser,
        max_retries=2
    )
    chat_messages = ChatPromptTemplate.from_messages([
        ('system', system_template),
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
    chain = chat_messages | doubao_llm | RunnableParallel(
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
            'result': image_result['json_result']
        }
    except Exception as e:
        print(e)
        return {
            'messages': messages,
            'result': None
        }


image_receipt_graph.add_node('vision_images_node', vision_images_node)

image_receipt_graph.add_edge(START, 'vision_images_node')
image_receipt_graph.add_edge('vision_images_node', END)

image_receipt_agent = image_receipt_graph.compile()
