import logging
import uuid

from fastapi import APIRouter, HTTPException, Request, status

from api.v1.schema.upload import UploadFileChunkResponse
from core.qdrant import delete_uploaded_documents
from services.v1.upload_service import upload_file_chunks

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload/chunks", response_model=UploadFileChunkResponse)
async def upload_chunks(request: Request):
    file_id = request.headers.get("x-file-id")
    file_name = request.headers.get("x-filename")
    chunk_index = int(request.headers.get("x-chunk-index", 0))
    total_chunks = int(request.headers.get("x-total-chunks", 1))

    if not file_name:
        raise HTTPException(status_code=400, detail="Filename required")

    if not file_id:
        file_id = str(uuid.uuid4())

    return await upload_file_chunks(
        request=request,
        file_id=file_id,
        file_name=file_name,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )


@router.delete("/upload/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_uploaded_file(file_id: str):
    delete_uploaded_documents({"file_id": file_id})
