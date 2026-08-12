import os
import csv
import math
import numpy as np
import pandas as pd

from src.search.embedding_engine import EmbeddingEngine


class ImageSearch:
    """
    Semantic and metadata-based image search using CLIP embeddings.

    Supports:
    - Text-to-image search
    - Image-to-image search
    - Metadata search by latitude, longitude and/or image ID

    Metadata search behavior:
    - ID only: returns the image with that exact ID (or filename/stem).
    - Latitude + longitude: returns geographically closest images.
    - Latitude only: returns images closest in latitude.
    - Longitude only: returns images closest in longitude.
    - ID + coordinates: restricts to the matching ID when found and
      ranks it by geographic distance.
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

        self.metadata_path = os.path.join(
            "data",
            "raw",
            "metadata.csv"
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

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
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
        # Load metadata
        # -------------------------------------------------

        self.metadata = pd.read_csv(
            self.metadata_path
        )

        required_columns = {
            "id",
            "latitude",
            "longitude"
        }

        missing_columns = (
            required_columns
            - set(self.metadata.columns)
        )

        if missing_columns:
            raise ValueError(
                "Metadata file is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        self.metadata["id"] = (
            self.metadata["id"]
            .fillna("")
            .astype(str)
            .str.replace("\\", "/", regex=False)
            .str.lstrip("./")
        )

        self.metadata["latitude"] = pd.to_numeric(
            self.metadata["latitude"],
            errors="coerce"
        )

        self.metadata["longitude"] = pd.to_numeric(
            self.metadata["longitude"],
            errors="coerce"
        )

        # Fast lookup by metadata ID.
        self.metadata_by_id = {}

        for _, row in self.metadata.iterrows():

            metadata_id = self._normalise_id(
                row["id"]
            )

            if metadata_id:
                self.metadata_by_id[
                    metadata_id
                ] = {
                    "id": metadata_id,
                    "latitude": (
                        float(row["latitude"])
                        if pd.notna(row["latitude"])
                        else None
                    ),
                    "longitude": (
                        float(row["longitude"])
                        if pd.notna(row["longitude"])
                        else None
                    )
                }

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

        print(
            f"Loaded metadata rows: {len(self.metadata)}"
        )

        print()
        print("IMAGE SEARCH READY")
        print("=" * 60)

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _normalise_id(value):
        """Normalise IDs/paths so metadata and index paths can match."""
        if value is None:
            return ""

        value = str(value).strip()

        if not value:
            return ""

        value = value.replace("\\", "/")
        value = value.lstrip("./")

        return value

    def _metadata_for_index(self, index):
        """
        Match an indexed image to its metadata row.

        The metadata ID contains the relative image path, e.g.
        f1/53/233371247.jpg, while image_ids.csv may contain only
        233371247. We therefore try the stored path first and then
        the filename/stem.
        """

        stored_path = ""

        if (
            index < len(self.image_paths)
            and self.image_paths[index]
        ):
            stored_path = self._normalise_id(
                self.image_paths[index]
            )

            # Make path relative to data/raw/images when possible.
            marker = "data/raw/images/"
            if marker in stored_path.lower():

                position = stored_path.lower().find(
                    marker
                )

                stored_path = stored_path[
                    position + len(marker):
                ]

        if stored_path:
            metadata = self.metadata_by_id.get(
                stored_path
            )

            if metadata:
                return metadata

        # Try the image ID as a complete metadata ID.
        image_id = self._normalise_id(
            self.image_ids[index]
            if index < len(self.image_ids)
            else ""
        )

        metadata = self.metadata_by_id.get(
            image_id
        )

        if metadata:
            return metadata

        # Finally match by filename/stem.
        image_filename = os.path.basename(
            stored_path or image_id
        )

        image_stem = os.path.splitext(
            image_filename
        )[0].lower()

        for metadata_id, metadata in self.metadata_by_id.items():

            metadata_filename = os.path.basename(
                metadata_id
            )

            metadata_stem = os.path.splitext(
                metadata_filename
            )[0].lower()

            if (
                image_filename.lower()
                == metadata_filename.lower()
                or image_stem == metadata_stem
            ):
                return metadata

        return None

    @staticmethod
    def _haversine_km(
        latitude1,
        longitude1,
        latitude2,
        longitude2
    ):
        """Return great-circle distance between two coordinates in km."""

        lat1 = math.radians(latitude1)
        lon1 = math.radians(longitude1)
        lat2 = math.radians(latitude2)
        lon2 = math.radians(longitude2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1)
            * math.cos(lat2)
            * math.sin(dlon / 2) ** 2
        )

        return (
            6371.0088
            * 2
            * math.atan2(
                math.sqrt(a),
                math.sqrt(1 - a)
            )
        )

    # =====================================================
    # TEXT SEARCH
    # =====================================================

    def search(
        self,
        query,
        top_k=5
    ):

        query_embedding = self.engine.encode_text(
            query
        )

        similarities = np.dot(
            self.embeddings,
            query_embedding
        )

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

        query_embedding = self.engine.encode_image(
            image_path
        )

        similarities = np.dot(
            self.embeddings,
            query_embedding
        )

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
    # METADATA SEARCH
    # =====================================================

    def search_by_metadata(
        self,
        latitude=None,
        longitude=None,
        image_id=None,
        top_k=1
    ):
        """
        Search dataset images using any combination of:

            latitude
            longitude
            image_id

        Examples:

        1. latitude + longitude
           -> geographically closest images

        2. latitude only
           -> closest latitude

        3. longitude only
           -> closest longitude

        4. image_id only
           -> exact matching image

        5. image_id + latitude/longitude
           -> matching ID is prioritised and ranked by
              coordinate distance when coordinates are supplied.
        """

        if (
            latitude is None
            and longitude is None
            and image_id is None
        ):
            raise ValueError(
                "Provide at least one of latitude, longitude or id."
            )

        # -------------------------------------------------
        # Validate coordinates
        # -------------------------------------------------

        if latitude is not None:

            latitude = float(latitude)

            if not -90 <= latitude <= 90:
                raise ValueError(
                    "Latitude must be between -90 and 90."
                )

        if longitude is not None:

            longitude = float(longitude)

            if not -180 <= longitude <= 180:
                raise ValueError(
                    "Longitude must be between -180 and 180."
                )

        # -------------------------------------------------
        # Find exact ID candidates
        # -------------------------------------------------

        id_candidates = None

        if image_id is not None:

            image_id = self._normalise_id(
                image_id
            )

            if not image_id:
                raise ValueError(
                    "ID cannot be empty."
                )

            exact_matches = []

            # Match the complete metadata ID.
            if image_id in self.metadata_by_id:
                exact_matches.append(
                    image_id
                )

            # Also allow entering just the filename or
            # numeric stem, e.g. 233371247.jpg / 233371247.
            entered_filename = os.path.basename(
                image_id
            )

            entered_stem = os.path.splitext(
                entered_filename
            )[0].lower()

            if not exact_matches:

                for metadata_id in self.metadata_by_id:

                    metadata_filename = os.path.basename(
                        metadata_id
                    )

                    metadata_stem = os.path.splitext(
                        metadata_filename
                    )[0].lower()

                    if (
                        entered_filename.lower()
                        == metadata_filename.lower()
                        or entered_stem
                        == metadata_stem
                    ):
                        exact_matches.append(
                            metadata_id
                        )

            if exact_matches:
                id_candidates = set(
                    exact_matches
                )
            else:
                raise ValueError(
                    f"No image found with id '{image_id}'."
                )

        # -------------------------------------------------
        # Score every indexed image
        # -------------------------------------------------

        candidates = []

        for index in range(
            min(
                len(self.embeddings),
                len(self.image_ids)
            )
        ):

            metadata = self._metadata_for_index(
                index
            )

            if not metadata:
                continue

            metadata_id = metadata["id"]

            # If ID was supplied, only matching image(s)
            # are considered.
            if (
                id_candidates is not None
                and metadata_id not in id_candidates
            ):
                continue

            image_latitude = metadata["latitude"]
            image_longitude = metadata["longitude"]

            distance_km = None
            coordinate_distance = None

            # -------------------------------------------------
            # Both latitude and longitude
            # -------------------------------------------------

            if (
                latitude is not None
                and longitude is not None
            ):

                if (
                    image_latitude is None
                    or image_longitude is None
                ):
                    continue

                distance_km = self._haversine_km(
                    latitude,
                    longitude,
                    image_latitude,
                    image_longitude
                )

                coordinate_distance = distance_km

            # -------------------------------------------------
            # Latitude only
            # -------------------------------------------------

            elif latitude is not None:

                if image_latitude is None:
                    continue

                coordinate_distance = abs(
                    latitude - image_latitude
                )

            # -------------------------------------------------
            # Longitude only
            # -------------------------------------------------

            elif longitude is not None:

                if image_longitude is None:
                    continue

                # Simple longitude difference. This is
                # appropriate for a longitude-only search.
                coordinate_distance = abs(
                    longitude - image_longitude
                )

            # -------------------------------------------------
            # ID only
            # -------------------------------------------------

            else:

                coordinate_distance = 0.0

            image_path = self._find_image(
                self.image_ids[index],
                index
            )

            candidates.append({
                "index": index,
                "image_id": metadata_id,
                "image_path": image_path,
                "latitude": image_latitude,
                "longitude": image_longitude,
                "distance_km": (
                    float(distance_km)
                    if distance_km is not None
                    else None
                ),
                "_sort_distance": float(
                    coordinate_distance
                )
            })

        if not candidates:

            raise ValueError(
                "No indexed image has matching metadata."
            )

        candidates.sort(
            key=lambda item: item["_sort_distance"]
        )

        results = []

        for rank, result in enumerate(
            candidates[:top_k],
            start=1
        ):

            result.pop(
                "_sort_distance",
                None
            )

            result["rank"] = rank

            # Frontend-ready URL for the FastAPI /images mount.
            image_path = result.get("image_path")

            if image_path:
                normalised_path = image_path.replace(
                    "\\",
                    "/"
                )

                marker = "data/raw/images/"

                if marker in normalised_path.lower():

                    position = normalised_path.lower().find(
                        marker
                    )

                    relative_image_path = normalised_path[
                        position + len(marker):
                    ]

                else:
                    relative_image_path = normalised_path

                result["image_url"] = (
                    "/images/"
                    + relative_image_path.lstrip("/")
                )
            else:
                result["image_url"] = None

            results.append(
                result
            )

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

                stored_path = stored_path.replace(
                    "\\",
                    "/"
                )

                stored_path = stored_path.lstrip(
                    "./\\"
                )

                # The index stores paths such as:
                # data/raw/images/f1/53/file.jpg
                if os.path.exists(stored_path):

                    return stored_path.replace(
                        "\\",
                        "/"
                    )

                # If the stored path is absolute or comes
                # from another machine, recover the part
                # after data/raw/images/.
                marker = "data/raw/images/"

                if marker in stored_path.lower():

                    position = stored_path.lower().find(
                        marker
                    )

                    relative_path = stored_path[
                        position + len(marker):
                    ]

                    full_path = os.path.join(
                        self.image_root,
                        relative_path
                    )

                    if os.path.exists(full_path):

                        return os.path.relpath(
                            full_path,
                            "."
                        ).replace(
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

        # First try metadata ID as a relative path.
        normalised_id = self._normalise_id(
            image_id
        )

        if normalised_id:

            direct_path = os.path.join(
                self.image_root,
                normalised_id
            )

            if os.path.isfile(direct_path):

                return os.path.relpath(
                    direct_path,
                    "."
                ).replace(
                    "\\",
                    "/"
                )

        # Finally search by filename.
        filename = os.path.basename(
            normalised_id
        )

        if not os.path.splitext(filename)[1]:
            filename = None

        for root, dirs, files in os.walk(
            self.image_root
        ):

            if filename and filename in files:

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