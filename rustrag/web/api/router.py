from fastapi.routing import APIRouter

from rustrag.web.api import echo, monitoring, rag

api_router = APIRouter()
api_router.include_router(monitoring.router)
api_router.include_router(echo.router, prefix="/echo", tags=["echo"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
