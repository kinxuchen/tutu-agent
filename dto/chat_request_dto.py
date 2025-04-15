from pydantic import BaseModel, Field
from typing import List, Union


class ChatRequestBody(BaseModel):
    input: str # 用户输入
    image_urls: Union[List[str],None] = Field(description='图片地址', default=None)
    is_create_order: bool = False # 是否需要创建订单
    is_resuming: bool = Field(description='是否中断', default=False)

