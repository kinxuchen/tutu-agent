import enum

# 中断任务类型
class RESUME_TYPE(enum.Enum):
    all = 0
    clientele = 1

# 当前的任务类型，判断是销售单还是入库单
class TASK_TYPE(enum.Enum):
    storage=0 # 入库单
    sell=1 # 散客单，批发单
