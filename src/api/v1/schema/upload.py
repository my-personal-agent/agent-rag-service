from pydantic import BaseModel


class UploadFileChunkResponse(BaseModel):
    file_id: str
    file_name: str
    complete: bool
