from langchain_milvus import Milvus, Zilliz
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from components.store import milvus_vector_store


example_selector = SemanticSimilarityExampleSelector(
    vectorstore=milvus_vector_store,
    k=2,
    input_keys=['input'],  # 根据 input_keys 查找相似度
    example_keys=['input', 'output'], # 从示例查询结果中提取的字段
    vectorstore_kwargs={
        ## langchain 和 milvus 的实现参数不一致，缺失 s
        "param": {
            "radius": 0.2,
        }
    }
)

example_prompt = PromptTemplate(
    input_variables=['input', 'output'],
    template="输入:{input}\n输出:{output}"
)

t = """
    ## 注意
    当你获取结果后，需要将获取的内容以 JSON 的结构返回。
    JSON 有一个顶层字段是 goods, goods 字段是一个数组，数组项的每一项是一个Dict字典结构，包含以下字段
    - name 字段表示名称
    - color 字段表示颜色
    - count 字段表示数量
    不要包含 Markdown 符号
    如果无法解析，请将 goods 解析成一个空数组
"""
few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    prefix="接下来我会提供一些示例，示例可能为空",
    suffix="""
        请根据所提供的示例，提取对应的数据结构，并选择对应的工具调用
        下面是用户的输入:{input}
        ## 注意
        - 如果没有相关示例，请返回空。只有存在示例的情况下，你才会思考用户的输入
        - 如果解析不出内容，可以返回空
        
        你可以使用下面这些工具
        {tool_names}
    """,
    input_variables=['input'],
    example_separator="\n\n\n",
)
