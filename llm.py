from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_deepseek import ChatDeepSeek
from constant import (
    BASE_URL,
    API_KEY,
    MODEL_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    IMAGE_API_KEY,
    IMAGE_BASE_URL,
    IMAGE_MODEL
)

llm = ChatOpenAI(
    streaming=True,
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    max_tokens=1024 * 10,
    max_retries=2
)

# 视觉识别的模型
image_llm = ChatOpenAI(
    streaming=True,
    model=IMAGE_MODEL,
    api_key=IMAGE_API_KEY,
    base_url=IMAGE_BASE_URL,
    max_retries=3
)

embeddings = OpenAIEmbeddings(
    api_key=EMBEDDING_API_KEY,
    base_url=EMBEDDING_BASE_URL,
    model=EMBEDDING_MODEL
)



ds_llm = ChatDeepSeek(
    model='deepseek-reasoner'
)
