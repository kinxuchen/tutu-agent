from typing import List, Union

from pydantic import BaseModel, Field

class SubGood(BaseModel):
    length: Union[float, int] = Field(description="商品 SKU 的具体长度")

# 基本的粗码字段
class BaseGoodResult(BaseModel):
    name: str = Field(description='商品名称, 必须解析出来')
    color: Union[str,None] = Field(description='商品颜色,解析不出可以为空字符串')
    count: Union[int,None] = Field(description='商品总件数,解析不出可以为0')
    length: Union[float, int, None] = Field(description="商品总米数, 解析不出可以为 0")
    clientele: Union[str, None] = Field(description='客户或者收货人信息,解析不出可以为空字符串')

# 细码
class SmallGoodsResult(BaseGoodResult):
    subitems: Union[List[SubGood], None] = Field(
        description="""
            尝试解析对应 SKU 每一件的具体的长度或者数量。
        """
    )


# 细码数据
class SmallGoodsResults(BaseModel):
    goods: List[SmallGoodsResult] = Field(description='商品列表')

# 粗码的数据
class ThickGoodsResults(BaseModel):
    goods: List[BaseGoodResult] = Field(description='商品列表')
