from pydantic import BaseModel, Field
from typing import List, Union


class ReceiptRequestBody(BaseModel):
    input: str = Field(description='用户输入的内容')
    image_urls: Union[List[str],None] = Field(description='图片地址', default=None)
    is_resuming: bool = Field(description='当前 Agent 调用是否是中断，不要任意传，接口会返回这个值', default=False)
    is_small: bool = Field(description="是否是细码", default=True)


# 普通的聊天输入
class ChatRequestBody(BaseModel):
    input: str = Field(description='用户输入的内容')


class ExampleRequestBody(BaseModel):
    input: str = Field(description='用户输入的内容')
    output: str = Field(description='用户输出的内容')
