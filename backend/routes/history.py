
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from ..schemas import PredictionOut
from ..services.db_service import get_user_history
from ..utils import get_current_user

router = APIRouter()

@router.get("/", response_model=list[PredictionOut])
def list_history(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    records = get_user_history(db, user_id=current_user.id)
    return [PredictionOut.from_orm(r) for r in records]
