from fastapi import APIRouter, UploadFile, HTTPException
from services.oss import oss_upload, cos_upload
from typing import List
from services.rag import reader_markdown_content

tool_router = APIRouter(prefix='/tools')


@tool_router.post('/upload/image')
async def upload_image_request(image: UploadFile):
    """上传图片的 OSS"""
    filename = image.filename
    try:
        result = await cos_upload(filename, image.file)
        return {
            "success": True,
            "url": result
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

