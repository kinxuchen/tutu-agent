from sqlalchemy import Column, String, Float, DateTime
from components.db import Base


class Inventory(Base):
    __tablename__ = "inventory"
    id = Column(
        String(36),
        primary_key=True,
        comment="主键id",
    )

    name = Column(
        String(255),
        comment="商品名称",
    )

    price = Column(
        Float,
        comment="价格",
    )

    len = Column(
        Float,
        comment='长度'
    )

    create_time = Column(
        DateTime,
        comment="创建时间",
    )

    update_time = Column(
        DateTime,
        comment="更新时间",
    )
    unit = Column(
        String(255),
        comment="单位",
    )
    color = Column(
        String(255),
        comment="颜色",
    )

    alias_name = Column(
        String(255),
        comment="商品别名",
    )
