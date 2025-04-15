from fastapi import APIRouter, UploadFile, HTTPException
from services.rag import reader_markdown_content, query_knowledge_server
from typing import List
from dto.rag_request import QuerySearch


rag_router = APIRouter(prefix='/rag')

@rag_router.post('/upload/md')
async def upload_markdown_request(markdowns: List[UploadFile]):
    try:
        await reader_markdown_content(files=markdowns)
        return {
            "success": True,
            "message": "加入知识库成功"
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@rag_router.post('/query')
async def query_request(body: QuerySearch):
    try:
        result = await query_knowledge_server(body.input)
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))
