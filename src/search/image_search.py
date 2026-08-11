import os
import csv
import numpy as np

from src.search.embedding_engine import EmbeddingEngine


class ImageSearch:
    """
    Semantic image search using CLIP embeddings.
    Supports:
    - Text-to-image search
    - Image-to-image search
    """

    def __init__(self):

        print()
        print("=" * 60)
        print("INITIALIZING IMAGE SEARCH")
        print("=" * 60)

        # -------------------------------------------------
        # Paths
        # -------------------------------------------------

        self.embeddings_path = os.path.join(
            "outputs",
            "search",
            "image_embeddings.npy"
        )

        self.ids_path = os.path.join(
            "outputs",
            "search",
            "image_ids.csv"
        )

        self.image_root = os.path.join(
            "data",
            "raw",
            "images"
        )

        # -------------------------------------------------
        # Check files
        # -------------------------------------------------

        if not os.path.exists(self.embeddings_path):
            raise FileNotFoundError(
                f"Embeddings file not found: {self.embeddings_path}"
            )

        if not os.path.exists(self.ids_path):
            raise FileNotFoundError(
                f"Image IDs file not found: {self.ids_path}"
            )

        # -------------------------------------------------
        # Load embeddings
        # -------------------------------------------------

        self.embeddings = np.load(
            self.embeddings_path
        )

        # -------------------------------------------------
        # Load image IDs and paths
        # -------------------------------------------------

        self.image_ids = []
        self.image_paths = []

        with open(
            self.ids_path,
            "r",
            encoding="utf-8",
            newline=""
        ) as f:

            reader = csv.reader(f)

            # Skip header
            next(reader, None)

            for row in reader:

                if not row:
                    continue

                image_id = row[0].strip()

                if not image_id:
                    continue

                self.image_ids.append(
                    image_id
                )

                if len(row) >= 2:
                    image_path = row[1].strip()
                else:
                    image_path = None

                self.image_paths.append(
                    image_path
                )

        # -------------------------------------------------
        # Load CLIP
        # -------------------------------------------------

        self.engine = EmbeddingEngine()

        print(
            f"Loaded embeddings: {self.embeddings.shape}"
        )

        print(
            f"Loaded image IDs: {len(self.image_ids)}"
        )

        print(
            f"Loaded image paths: {len(self.image_paths)}"
        )

        print()
        print("IMAGE SEARCH READY")
        print("=" * 60)

    # =====================================================
    # TEXT SEARCH
    # =====================================================

    def search(
        self,
        query,
        top_k=5
    ):

        # -------------------------------------------------
        # Convert text to CLIP embedding
        # -------------------------------------------------

        query_embedding = self.engine.encode_text(
            query
        )

        # -------------------------------------------------
        # Calculate cosine similarity
        # -------------------------------------------------

        similarities = np.dot(
            self.embeddings,
            query_embedding
        )

        # -------------------------------------------------
        # Get top results
        # -------------------------------------------------

        top_indices = np.argsort(
            similarities
        )[::-1][:top_k]

        results = []

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            image_id = self.image_ids[index]

            image_path = self._find_image(
                image_id,
                index
            )

            results.append({
                "rank": rank,
                "image_id": image_id,
                "image_path": image_path,
                "similarity": float(
                    similarities[index]
                )
            })

        return results

    # =====================================================
    # IMAGE SEARCH
    # =====================================================

    def search_by_image(
        self,
        image_path,
        top_k=5
    ):

        # -------------------------------------------------
        # Convert uploaded image to CLIP embedding
        # -------------------------------------------------

        query_embedding = self.engine.encode_image(
            image_path
        )

        # -------------------------------------------------
        # Calculate similarity
        # -------------------------------------------------

        similarities = np.dot(
            self.embeddings,
            query_embedding
        )

        # -------------------------------------------------
        # Get top results
        # -------------------------------------------------

        top_indices = np.argsort(
            similarities
        )[::-1][:top_k]

        results = []

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            image_id = self.image_ids[index]

            result_image_path = self._find_image(
                image_id,
                index
            )

            results.append({
                "rank": rank,
                "image_id": image_id,
                "image_path": result_image_path,
                "similarity": float(
                    similarities[index]
                )
            })

        return results

    # =====================================================
    # FIND IMAGE
    # =====================================================

    def _find_image(
        self,
        image_id,
        index=None
    ):

        # -------------------------------------------------
        # First try CSV path
        # -------------------------------------------------

        if (
            index is not None
            and index < len(self.image_paths)
        ):

            stored_path = self.image_paths[index]

            if stored_path:

                # Convert Windows backslashes
                stored_path = stored_path.replace(
                    "\\",
                    os.sep
                )

                # Remove possible leading ./ 
                stored_path = stored_path.lstrip(
                    "./\\"
                )

                full_path = stored_path

                if not os.path.isabs(full_path):

                    full_path = os.path.join(
                        "",
                        stored_path
                    )

                if os.path.exists(full_path):

                    return stored_path.replace(
                        "\\",
                        "/"
                    )

        # -------------------------------------------------
        # Search dataset folders
        # -------------------------------------------------

        filename_extensions = [
            ".jpg",
            ".jpeg",
            ".png"
        ]

        for root, dirs, files in os.walk(
            self.image_root
        ):

            for extension in filename_extensions:

                filename = (
                    f"{image_id}{extension}"
                )

                if filename in files:

                    full_path = os.path.join(
                        root,
                        filename
                    )

                    return os.path.relpath(
                        full_path,
                        "."
                    ).replace(
                        "\\",
                        "/"
                    )

        return None