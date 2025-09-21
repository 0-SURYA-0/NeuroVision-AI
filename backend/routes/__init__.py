
from fastapi import APIRouter
from .auth import router as auth_router
from .predict import router as predict_router
from .history import router as history_router

api_router = APIRouter()

api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(predict_router, prefix="/predict", tags=["predict"])
api_router.include_router(history_router, prefix="/history", tags=["history"])
