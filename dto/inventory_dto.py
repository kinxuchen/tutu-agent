from pydantic import BaseModel, Field

class InventoryDTO(BaseModel):
    name: str = Field(description="商品名称")
    price: float = Field(description="价格")
    len: float = Field(description="长度")
    create_time: str = Field(description="创建时间")
    update_time: str = Field(description="更新时间")
    color: str = Field(description="颜色")
    unit: str = Field(description="单位")
