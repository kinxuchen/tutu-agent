from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from constant import BASE_URL, API_KEY, MODEL_NAME

llm = ChatOpenAI(
    streaming=True,
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL
)


ds_llm = ChatDeepSeek(
    model='deepseek-reasoner'
)
