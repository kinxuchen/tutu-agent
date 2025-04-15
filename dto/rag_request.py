from pydantic import BaseModel, Field

class QuerySearch(BaseModel):
    input: str = Field(description='查询内容', default="")
