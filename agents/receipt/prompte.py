from langchain_core.prompts import ChatPromptTemplate


clientele_search_prompt = ChatPromptTemplate.from_messages(
    [
        ('user', """
            请提取我下面输入中的机构或者人名信息
            {input}
            你可以使用下面的工具
            {tool_names}
        """)
    ]
)