from sqlalchemy import Column, String, Integer, Enum
from components.db import Base
import enum

class GenderEnum(enum.Enum):
    MALE = "male"
    FEMALE = "female"


class Clientele(Base):
    __tablename__ = "clientele"
    id = Column(
        String(36),
        primary_key=True,
        comment="主键id",
    )
    name = Column(
        String(255),
        comment="客户姓名",
    )
    age = Column(
        Integer,
        comment="客户年龄",
    )
    gender = Column(
        Enum(GenderEnum),
        comment="客户性别",
        nullable=True
    )
