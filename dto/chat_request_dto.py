from pydantic import BaseModel, Field
from typing import List, Union


class ReceiptRequestBody(BaseModel):
    input: str # 用户输入
    image_urls: Union[List[str],None] = Field(description='图片地址', default=None)
    is_create_order: bool = False # 是否需要创建订单
    is_resuming: bool = Field(description='是否中断', default=False)
    is_small: bool = Field(description="是否是细码", default=True)

