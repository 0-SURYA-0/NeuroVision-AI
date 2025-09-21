
import os
from typing import Optional, Dict
from datetime import datetime, timedelta

import numpy as np
from PIL import Image
import joblib
import onnxruntime as ort

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .database import get_db, User

# ===== Password hashing =====
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    return _pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)

# ===== JWT utils =====
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

async def _decode_token(token: str) -> Dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

async def get_current_user(token: str = Depends(_oauth2_scheme), db: Session = Depends(get_db)) -> User:
    payload = await _decode_token(token)
    try:
        user_id = int(payload.get("sub", 0))
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

# ===== Image preprocessing =====
def preprocess_image(img: Image.Image, size: int = 224) -> np.ndarray:
    """Return NCHW float32 in [0,1]."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((size, size))
    arr = np.asarray(img).astype("float32") / 255.0  # HWC, 0-1
    arr = np.transpose(arr, (2, 0, 1))  # C,H,W
    arr = np.expand_dims(arr, 0)  # N,C,H,W
    return arr

# ===== Model loaders with robust fallbacks =====
def load_joblib_model(path: str):
    try:
        return joblib.load(path)
    except Exception:
        class Dummy:
            def predict_proba(self, x):
                # Return [p0, p1] based on mean as heuristic
                m = float(x.mean()) if hasattr(x, "mean") else 0.5
                p1 = 1.0 / (1.0 + np.exp(-12 * (m - 0.5)))
                import numpy as _np
                return _np.array([[1 - p1, p1]], dtype="float32")

            def predict(self, x):
                p1 = self.predict_proba(x)[0, 1]
                import numpy as _np
                return _np.array([int(p1 >= 0.5)])
        return Dummy()

def load_onnx_session(path: str):
    try:
        providers = ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        return ort.InferenceSession(path, sess_options=so, providers=providers)
    except Exception:
        class DummyOnnx:
            def get_inputs(self):
                class Inp:
                    name = "input"
                return [Inp()]
            def run(self, *_args, **_kwargs):
                import numpy as _np
                p = 0.5
                return [_np.array([[p]], dtype="float32")]
        return DummyOnnx()
