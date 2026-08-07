from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import shutil
import os

from src.predictor import AlbumPredictor


app = FastAPI(
    title="Image Album API",
    description="Predict which album an uploaded image belongs to",
    version="1.0"
)


# -----------------------------
# CORS Configuration
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Load ML Model
# -----------------------------

predictor = AlbumPredictor()


# -----------------------------
# Upload Folder
# -----------------------------

UPLOAD_FOLDER = os.path.join("outputs", "uploads")

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# -----------------------------
# Home Route
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to Image Album API"
    }


# -----------------------------
# Prediction Endpoint
# -----------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    try:

        # Save uploaded image temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )


        # Run AI prediction
        result = predictor.predict(
            file_path
        )


        # Handle model errors
        if "error" in result and result["error"]:
            raise HTTPException(
                status_code=400,
                detail=result["error"]
            )


        return result


    except HTTPException:
        raise


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    finally:

        # Remove temporary image
        if os.path.exists(file_path):
            os.remove(file_path)