from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

knowledge_search_prompts = ChatPromptTemplate.from_messages(
    [
        ('system', """
            ## 职责
            1. 根据用户的输入，从知识库提取相关知识
            ## 工具
            你可以使用下面这些工具:
            {tool_names}
        """),
        (
            'user', '{input}'
        )
    ]
)
