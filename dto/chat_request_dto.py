from pydantic import BaseModel


class ChatRequestBody(BaseModel):
    input: str # 用户输入
    is_create_order: bool = False # 是否需要创建订单
