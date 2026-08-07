import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.feature_extractor import FeatureExtractor
class AlbumPredictor:
    def __init__(
        self,
        cluster_csv="outputs/cluster_results.csv",
        images_folder="data/raw/images"
    ):
        self.extractor = FeatureExtractor()
        self.images_folder = images_folder
        self.cluster_df = pd.read_csv(cluster_csv)
        self.dataset_cache = []

        self._preload_dataset_features()

    def _preload_dataset_features(self):
        for _, row in self.cluster_df.iterrows():
            image_path = str(row["image_path"])
            cluster = row["cluster"]

            # Normalize slash direction for cross-platform compatibility
            normalized_path = image_path.replace("\\", "/")
            if "/images/" in normalized_path:
                relative_path = normalized_path.split("/images/")[-1]
            else:
                relative_path = os.path.basename(normalized_path)

            local_path = os.path.join(
                self.images_folder,
                *relative_path.split("/")
            )

            if not os.path.exists(local_path):
                continue

            try:
                feature = self.extractor.extract(local_path)
                self.dataset_cache.append({
                    "cluster": cluster,
                    "filename": os.path.basename(local_path),
                    "feature": feature
                })
            except Exception:
                continue

    def predict(self, uploaded_image_path):
        uploaded_feature = self.extractor.extract(uploaded_image_path)

        if not self.dataset_cache:
            return {
                "error": "No valid dataset images were loaded or found. Check images_folder path.",
                "album": None,
                "cluster": None,
                "matched_image": None,
                "similarity": 0.0
            }

        best_score = -1.0
        best_cluster = None
        best_image = None

        for item in self.dataset_cache:
            score = cosine_similarity(
                uploaded_feature.reshape(1, -1),
                item["feature"].reshape(1, -1)
            )[0][0]

            if score > best_score:
                best_score = score
                best_cluster = item["cluster"]
                best_image = item["filename"]

        if best_cluster is None:
            return {
                "error": "Could not determine similarity.",
                "album": None,
                "cluster": None,
                "matched_image": None,
                "similarity": 0.0
            }

        return {
            "album": f"Album_{best_cluster}",
            "cluster": int(best_cluster),
            "matched_image": best_image,
            "similarity": round(float(best_score) * 100, 2)
        }