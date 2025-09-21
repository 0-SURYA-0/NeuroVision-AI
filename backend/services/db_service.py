
from sqlalchemy.orm import Session
from datetime import datetime
from ..database import User, Prediction
from ..schemas import UserCreate
from ..utils import get_password_hash

# Users
def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, payload: UserCreate):
    user = User(
        email=payload.email,
        hashed_password=get_password_hash(payload.password),
        created_at=datetime.utcnow(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

# Predictions
def create_prediction(db: Session, user_id: int, filename: str, model_type: str, probability: float, prediction: int):
    rec = Prediction(
        user_id=user_id,
        filename=filename,
        model_type=model_type,
        probability=probability,
        prediction=prediction,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec

def get_user_history(db: Session, user_id: int):
    return (
        db.query(Prediction)
        .filter(Prediction.user_id == user_id)
        .order_by(Prediction.created_at.desc())
        .all()
    )
