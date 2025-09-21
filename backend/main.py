from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .database import Base, engine
from .routes import api_router

# Create DB tables at startup (safe if already exist)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Brain Tumor Detection API",
    version="1.0.0",
    description="Inference API with JWT auth, prediction history, and ONNX/joblib backends."
)

# CORS (adjust for your frontend origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(api_router)

@app.get("/", tags=["health"])
async def root():
    return {"status": "ok"}

@app.post("/predict-3d", tags=["3D Segmentation"])
async def predict_3d(file: UploadFile):
    """
    Endpoint to handle 3D .nii.gz file uploads for tumor segmentation.
    """
    return {"message": "3D prediction route is under construction."}
