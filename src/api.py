from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import shutil
import os

from src.predictor import AlbumPredictor
from src.search.image_search import ImageSearch


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Album Creator API",
    description="AI-powered image album prediction and semantic image search",
    version="1.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",

        # Your Vite frontend may use this port
        "http://localhost:5175",
        "http://127.0.0.1:5175"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "uploads"
)

DATASET_IMAGES_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "images"
)


# Create upload directory if it does not exist
os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# SERVE DATASET IMAGES
# ============================================================

# This is the important part for displaying
# the similar images in the React frontend.

if os.path.exists(DATASET_IMAGES_FOLDER):

    app.mount(
        "/images",
        StaticFiles(
            directory=DATASET_IMAGES_FOLDER
        ),
        name="images"
    )

    print(
        f"Dataset images available at: "
        f"http://127.0.0.1:8000/images/"
    )

else:

    print(
        "WARNING: Dataset image directory not found:"
    )

    print(
        DATASET_IMAGES_FOLDER
    )


# ============================================================
# LOAD AI MODELS
# ============================================================

print()
print("=" * 60)
print("LOADING AI MODELS")
print("=" * 60)

print()
print("Loading Album Predictor...")

predictor = AlbumPredictor()

print(
    "Album Predictor loaded successfully."
)


print()
print("Loading Image Search Engine...")

image_search = ImageSearch()

print(
    "Image Search Engine loaded successfully."
)


print()
print("=" * 60)
print("ALL AI SYSTEMS READY")
print("=" * 60)


# ============================================================
# HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Welcome to AI Album Creator API",

        "status": "running",

        "services": [
            "album prediction",
            "text image search",
            "image-to-image search"
        ]
    }


# ============================================================
# PREDICT ALBUM FROM IMAGE
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    try:

        # ----------------------------------------------------
        # Save uploaded image
        # ----------------------------------------------------

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # ----------------------------------------------------
        # Run album prediction
        # ----------------------------------------------------

        result = predictor.predict(
            file_path
        )


        # ----------------------------------------------------
        # Handle prediction errors
        # ----------------------------------------------------

        if (
            isinstance(result, dict)
            and result.get("error")
        ):

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

        # ----------------------------------------------------
        # Delete temporary uploaded image
        # ----------------------------------------------------

        if os.path.exists(file_path):

            os.remove(file_path)


# ============================================================
# TEXT-TO-IMAGE SEARCH
# ============================================================

@app.get("/search")
def search_images(

    query: str = Query(
        ...,
        description="Natural language search query"
    ),

    top_k: int = Query(
        5,
        ge=1,
        le=20,
        description="Number of results to return"
    )
):

    try:

        # ----------------------------------------------------
        # Validate query
        # ----------------------------------------------------

        if not query.strip():

            raise HTTPException(
                status_code=400,
                detail="Search query cannot be empty."
            )


        # ----------------------------------------------------
        # Search image index
        # ----------------------------------------------------

        results = image_search.search(
            query,
            top_k
        )


        return {
            "query": query,
            "results": results
        }


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# IMAGE-TO-IMAGE SEARCH
# ============================================================

@app.post("/search-by-image")
async def search_by_image(

    file: UploadFile = File(...),

    top_k: int = Query(
        5,
        ge=1,
        le=20,
        description="Number of similar images to return"
    )
):

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    try:

        # ----------------------------------------------------
        # Save uploaded image temporarily
        # ----------------------------------------------------

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # ----------------------------------------------------
        # Search using uploaded image
        # ----------------------------------------------------

        results = image_search.search_by_image(
            file_path,
            top_k
        )


        # ----------------------------------------------------
        # Return results
        # ----------------------------------------------------

        return {
            "query_type": "image",
            "results": results
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    finally:

        # ----------------------------------------------------
        # Delete temporary uploaded image
        # ----------------------------------------------------

        if os.path.exists(file_path):

            os.remove(file_path)


# ============================================================
# METADATA-BASED IMAGE SEARCH
# ============================================================

@app.get("/search-by-metadata")
def search_by_metadata(
    latitude: float | None = Query(
        None,
        description="Latitude to search near (-90 to 90)"
    ),

    longitude: float | None = Query(
        None,
        description="Longitude to search near (-180 to 180)"
    ),

    id: str | None = Query(
        None,
        description="Image metadata ID/path, filename, or filename stem"
    ),

    top_k: int = Query(
        1,
        ge=1,
        le=20,
        description="Number of closest images to return"
    )
):
    """
    Search images using metadata from data/raw/metadata.csv.

    You can provide any one, any two, or all three:
    latitude, longitude and id.

    Examples:
    - /search-by-metadata?latitude=40.71455&longitude=-74.007118
    - /search-by-metadata?latitude=40.71455
    - /search-by-metadata?longitude=-74.007118
    - /search-by-metadata?id=38/e2/6343684939.jpg
    - /search-by-metadata?id=6343684939
    - /search-by-metadata?latitude=40.71455&id=6343684939
    """

    if (
        latitude is None
        and longitude is None
        and id is None
    ):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of latitude, longitude or id."
        )

    try:

        results = image_search.search_by_metadata(
            latitude=latitude,
            longitude=longitude,
            image_id=id,
            top_k=top_k
        )

        return {
            "query_type": "metadata",
            "query": {
                "latitude": latitude,
                "longitude": longitude,
                "id": id
            },
            "results": results
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "api": "running",
        "album_predictor": "loaded",
        "image_search": "loaded"
    }