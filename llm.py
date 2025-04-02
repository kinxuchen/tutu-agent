from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_deepseek import ChatDeepSeek
from constant import BASE_URL, API_KEY, MODEL_NAME, EMBEDDING_MODEL, EMBEDDING_API_KEY, EMBEDDING_BASE_URL

llm = ChatOpenAI(
    streaming=True,
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL
)

embeddings = OpenAIEmbeddings(
    api_key=EMBEDDING_API_KEY,
    base_url=EMBEDDING_BASE_URL,
    model=EMBEDDING_MODEL
)



ds_llm = ChatDeepSeek(
    model='deepseek-reasoner'
)
