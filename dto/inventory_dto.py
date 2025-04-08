from pydantic import BaseModel, Field
from typing import Optional, Sequence, List

class InventoryDTO(BaseModel):
    id: str = Field(description="商品id", default=None)
    name: str = Field(description="商品名称")
    price: float = Field(description="价格")
    len: float = Field(description="长度")
    create_time: str = Field(description="创建时间")
    update_time: str = Field(description="更新时间")
    color: str = Field(description="颜色")
    unit: str = Field(description="单位")
    alias_name: Optional[str] = Field(description="商品别名", default=None)


# 客户信息
class ClienteleDTO(BaseModel):
    id: str = Field(description="客户id", default=None)
    name: str = Field(description="客户姓名")
    age: int = Field(description="年龄")
    gender: str = Field(description="性别")


class EmbeddingDTO(BaseModel):
    text: str = Field(description="文本")

# 示例输出
class ExampleDTO(BaseModel):
    input: str = Field(description="输入")
    output: str = Field(description="输出")

class ExampleListDTO(BaseModel):
    examples: List[ExampleDTO] = Field(description="示例列表")
