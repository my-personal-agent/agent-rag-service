import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
from fastapi import HTTPException, Request
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from api.v1.schema.upload import UploadFileChunkResponse
from config.settings_config import get_settings
from core.qdrant import add_documents_to_qdrant, delete_uploaded_documents

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".txt": TextLoader,
    ".pdf": PyMuPDFLoader,
    ".csv": CSVLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".json": JSONLoader,
}


def get_loader(file_path: str, file_extension: str):
    """Get appropriate loader for file type"""
    loader_class = SUPPORTED_EXTENSIONS.get(file_extension.lower())
    if not loader_class:
        raise ValueError(f"Unsupported file type: {file_extension}")

    # Special handling for JSON files
    if file_extension.lower() == ".json":
        return loader_class(file_path, jq_schema=".", text_content=False)

    return loader_class(file_path)


def process_file(
    file_id: str, file_path: str, metadata: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """Process a file and return chunked documents"""
    file_extension = Path(file_path).suffix

    loader = get_loader(file_path, file_extension)
    documents = loader.load()

    # Add metadata to documents
    for doc in documents:
        doc.metadata.update(metadata or {})
        doc.metadata["file_id"] = file_id
        doc.metadata["file_name"] = Path(file_path).name
        doc.metadata["file_extension"] = file_extension
        doc.metadata["processed_at"] = datetime.now().isoformat()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    # Add chunk metadata
    for i, split in enumerate(chunks):
        split.metadata["chunk_index"] = i
        split.metadata["total_chunks"] = len(chunks)

    return chunks


async def upload_file_chunks(
    request: Request,
    file_id: str,
    file_name: str,
    chunk_index: int,
    total_chunks: int,
) -> UploadFileChunkResponse:
    temp_dir = os.path.join(get_settings().upload_temp_dir, file_id)
    chunks_dir = os.path.join(temp_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_index}")

    # Save each chunk
    async with aiofiles.open(chunk_path, "wb") as f:
        async for chunk in request.stream():
            await f.write(chunk)
        await f.flush()

    # If it's the last chunk, merge them all
    if chunk_index == total_chunks - 1:
        final_path = os.path.join(temp_dir, file_name)

        async with aiofiles.open(final_path, "wb") as outfile:
            for i in range(total_chunks):
                chunk_file = os.path.join(chunks_dir, f"chunk_{i}")
                if not os.path.exists(chunk_file):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing chunk {i}. Upload incomplete.",
                    )

                async with aiofiles.open(chunk_file, "rb") as infile:
                    content = await infile.read()
                    await outfile.write(content)
                    await outfile.flush()

                # os.remove(chunk_file)

        try:
            documents = process_file(file_id, final_path)
            add_documents_to_qdrant(documents)
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            raise HTTPException(status_code=500, detail="File processing failed")
        # finally:
        #     shutil.rmtree(temp_dir, ignore_errors=True)

        return UploadFileChunkResponse(
            file_name=file_name,
            file_id=file_id,
            complete=True,
        )

    return UploadFileChunkResponse(file_name=file_name, file_id=file_id, complete=False)


def delete_uploaded_file(file_id: str) -> None:
    delete_uploaded_documents({"file_id": file_id})
