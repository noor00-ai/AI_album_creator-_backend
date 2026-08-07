# AI Album Creator

An end-to-end intelligent image categorization and RESTful prediction engine. This project processes raw image collections using custom spatial feature extraction, groups visually similar images into albums, and provides a FastAPI service for real-time image classification using cosine similarity.


## Features

- Image preprocessing using OpenCV
- Automatic RGB color conversion
- Image resizing to a uniform resolution (128 × 128)
- Pixel normalization for feature extraction
- High-dimensional feature vector generation
- Cosine similarity-based image matching
- Fast in-memory caching for low-latency predictions
- REST API built with FastAPI
- Interactive API documentation with Swagger UI and ReDoc
- Cross-platform compatibility (Windows, Linux, macOS)

---

## Technologies Used

- Python
- FastAPI
- OpenCV
- NumPy
- Pandas
- scikit-learn
- Uvicorn

---

## Project Structure

```text
AI_Album_Creator/
│
├── data/
│   └── raw/
│       ├── images/                 # Raw image dataset
│       └── metadata.csv            # Image metadata (optional)
│
├── outputs/
│   ├── albums/                     # Generated image albums
│   ├── uploads/                    # Uploaded images
│   └── cluster_results.csv         # Image-cluster mappings
│
├── src/
│   ├── __init__.py
│   ├── api.py                      # FastAPI application
│   ├── feature_extractor.py        # Image preprocessing pipeline
│   └── predictor.py                # Similarity prediction engine
│
├── requirements.txt
└── README.md
```

---

## Dataset

**Dataset Name:** Custom Image Album Dataset

**Source:** Local raw image repository / Kaggle image collection

**Storage Directory:**

```
data/raw/images/
```

**Output File:**

```
outputs/cluster_results.csv
```

### Important Columns

| Column | Description |
|---------|-------------|
| image_path | Path of the reference image |
| cluster | Cluster ID assigned to the image |

---

## Methodology

### 1. Data Preprocessing

Implemented in **feature_extractor.py**

- Images are loaded using `cv2.imread()`.
- BGR images are converted into RGB.
- Images are resized to **128 × 128** pixels.
- Pixel values are normalized from **0–255** to **0.0–1.0**.

---

### 2. Feature Engineering

Each image is transformed into a numerical representation.

- Image size: **128 × 128 × 3**
- Flattened into a **49,152-dimensional feature vector**
- Captures spatial color distribution for similarity comparison

---

### 3. Prediction Pipeline

Implemented in **predictor.py**

#### Startup Caching

When the API starts:

- Reads `cluster_results.csv`
- Loads every reference image
- Extracts feature vectors
- Stores vectors in memory for fast lookup

#### Image Prediction

For every uploaded image:

1. Extract feature vector.
2. Compare with all cached vectors.
3. Calculate Cosine Similarity.
4. Select the highest similarity score.
5. Return the corresponding cluster.

Cosine Similarity:

```
Similarity(A,B) = (A · B) / (||A|| × ||B||)
```

---

## Performance

| Metric | Value |
|---------|--------|
| Feature Vector Size | 49,152 values |
| Cached Inference Latency | <150 ms* |
| Target Similarity Score | >85%* |
| API Response Success Rate | 99.9%* |

\*Values shown are benchmark/target values and may vary depending on hardware and dataset size.

---

## API Endpoints

### GET /

Health check endpoint.

---

### POST /predict

Uploads an image and predicts its album cluster.

Example Response:

```json
{
    "cluster": 4,
    "similarity": 0.93,
    "image": "uploads/test.jpg"
}
```

---

### GET /docs

Interactive Swagger UI documentation.

```
http://127.0.0.1:8000/docs
```

---

### GET /redoc

ReDoc API documentation.

```
http://127.0.0.1:8000/redoc
```

---

## Requirements

- Python 3.10 or later
- pip
- Git

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/noor00-ai/AI_Album_Creator.git
cd AI_Album_Creator
```

---

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the FastAPI server:

```bash
uvicorn src.api:app --reload
```

Open your browser:

Swagger UI

```
http://127.0.0.1:8000/docs
```

ReDoc

```
http://127.0.0.1:8000/redoc
```

---

## Future Improvements

- Replace handcrafted feature vectors with deep learning embeddings.
- Integrate CNN-based image feature extraction.
- Support incremental album updates without rebuilding the cache.
- Add authentication and user management.
- Deploy the API using Docker and cloud platforms.
- Improve similarity search using FAISS or Annoy for large datasets.

---

## Author

**Noor Ul Ain**

GitHub: https://github.com/noor00-ai

---

## Acknowledgements

This project makes use of the following open-source libraries and tools:

- FastAPI
- OpenCV
- NumPy
- Pandas
- scikit-learn
- Uvicorn
- Kaggle image datasets

---

## License

This project is intended for educational and learning purposes. You may modify and use the code for personal or academic projects.