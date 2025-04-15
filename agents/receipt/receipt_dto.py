from typing import List, Union

from pydantic import BaseModel, Field


class GoodResult(BaseModel):
    name: str = Field(description='商品名称, 必须解析出来')
    color: Union[str,None] = Field(description='商品颜色,解析不出可以为空字符串')
    count: Union[int,None] = Field(description='商品总件数,解析不出可以为0')
    length: Union[float, int, None] = Field(description="商品总米数, 解析不出可以为 0")
    clientele: Union[str, None] = Field(description='客户或者收货人信息,解析不出可以为空字符串')


class GoodsResults(BaseModel):
    goods: List[GoodResult] = Field(description='商品列表')
