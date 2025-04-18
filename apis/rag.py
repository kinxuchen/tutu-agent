from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig

from services.rag import reader_markdown_content, query_knowledge_server
from typing import List
from dto.rag_request import QuerySearch
from agents.rag.agent import rag_agent, RagState
import jsonpickle

rag_router = APIRouter(prefix='/rag')

@rag_router.post('/upload/md')
async def upload_markdown_request(markdowns: List[UploadFile]):
    """上传 Markdown 文件到知识库"""
    try:
        await reader_markdown_content(files=markdowns)
        return {
            "success": True,
            "message": "加入知识库成功"
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@rag_router.post('/query/{thread_id}')
async def query_request(thread_id: str, body: QuerySearch):
    """Request/Response 的形式调用 RAG 查询"""
    try:
        config = {
            'configurable': {
                'thread_id': thread_id,
                'user_id': 'user_id_123'
            }
        }
        search_result = await rag_agent.ainvoke(input={
            'input': body.input,
            'urls': []
        }, config=config)
        return {
            'success': True,
            'content': search_result['messages'][-1].content,
            'urls': search_result['urls']
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@rag_router.post('/stream/{thread_id}')
async def stream__request(thread_id: str, body: QuerySearch):
    """以流式响应的方式进行 RAG 查询"""
    from langchain_core.messages import AIMessageChunk, AIMessage
    config = {
        'configurable': {
            'thread_id': thread_id,
            'user_id': 'user_id_123'
        }
    }

    async def generator_stream():
        yield f"data:{jsonpickle.encode({
            'status': 'stared'
        })}\n\n"
        try:
            async for msg,metadata in rag_agent.astream({
                'input': body.input,
                'urls': []
            }, config=config, stream_mode="messages"):
                if msg and isinstance(msg, AIMessageChunk):
                    yield f"data:{jsonpickle.encode({
                        'content': msg.content
                    })}\n\n"
            yield f"data: {jsonpickle.encode({
                'status': 'completed',
                'urls': rag_agent.get_state(config).values['urls']
            })}\n\n"
        except Exception as e:
            yield f"data:{jsonpickle.encode({
                'status': 'error',
                'message': str(e)
            })}\n\n"
    return StreamingResponse(
        generator_stream(),
        media_type="text/event-stream"
    )

