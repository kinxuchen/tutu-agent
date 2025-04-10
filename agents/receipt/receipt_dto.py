from typing import List

from pydantic import BaseModel, Field


class GoodResult(BaseModel):
    name: str = Field(description='商品名称, 必须解析出来')
    color: str = Field(description='颜色,解析不出可以为空字符串')
    count: int = Field(description='数量,解析不出可以为0')
    clientele: str = Field(description='客户信息,解析不出可以为空字符串')


class GoodsResults(BaseModel):
    goods: List[GoodResult] = Field(description='商品列表')
