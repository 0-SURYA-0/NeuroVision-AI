from pydantic import BaseModel, EmailStr, Field
from datetime import datetime

# Auth
class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str = Field(min_length=6)

class UserLogin(UserBase):
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

# Predictions
class PredictionBase(BaseModel):
    filename: str
    model_type: str
    probability: float
    prediction: int

class PredictionOut(PredictionBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True  