"""处理图片识别相关的 Agent """
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List, Any
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from llm import image_llm, llm, doubao_llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import RetryOutputParser, OutputFixingParser
from agents.receipt.receipt_dto import SmallGoodsResults,ThickGoodsResults
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from agents.receipt.prompte import small_system_template, thick_system_template


class ImageReceiptState(BaseModel):
    image_urls: List[str] = Field(description='图片地址', default=None)
    messages: List[BaseMessage] = Field(description='消息列表', default=[])
    result: Any = Field(description='生成单据的结果', default=None)
    is_small: bool = Field(description="是否是细码", default=True)
    # todo 需要加一个字段，表示当前是粗码还是细码


image_receipt_graph = StateGraph(ImageReceiptState)

def vision_images_node(state: ImageReceiptState):
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
