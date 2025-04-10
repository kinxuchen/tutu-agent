from pydantic import BaseModel, Field


class ChatRequestBody(BaseModel):
    input: str # 用户输入
    is_create_order: bool = False # 是否需要创建订单
    is_resuming: bool = Field(description='是否中断', default=False)
