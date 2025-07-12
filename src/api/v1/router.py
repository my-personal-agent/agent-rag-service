from fastapi import APIRouter

from api.v1.endpoints.upload import router as upload_router

api_router = APIRouter()

api_router.include_router(upload_router)
