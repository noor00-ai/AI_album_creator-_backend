import os
import csv
import numpy as np

from src.search.embedding_engine import EmbeddingEngine

IMAGE_ROOT = "data/raw/images"
OUTPUT_DIR = "outputs/search"

EMBEDDINGS_FILE = os.path.join(
    OUTPUT_DIR,
    "image_embeddings.npy"
)

IDS_FILE = os.path.join(
    OUTPUT_DIR,
    "image_ids.csv"
)


def find_images():

    images = []

    for root, _, files in os.walk(IMAGE_ROOT):

        for filename in files:

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                full_path = os.path.join(
                    root,
                    filename
                )

                images.append(full_path)

    return images


def main():

    print("=" * 60)
    print("AI ALBUM CREATOR - BUILD SEARCH INDEX")
    print("=" * 60)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    image_paths = find_images()

    print()
    print("Images found:", len(image_paths))
    print()

    if not image_paths:
        print("No images found.")
        return

    engine = EmbeddingEngine()

    embeddings = []
    records = []

    total = len(image_paths)

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        try:

            embedding = engine.encode_image(
                image_path
            )

            embeddings.append(
                embedding
            )

            image_id = os.path.splitext(
                os.path.basename(image_path)
            )[0]

            records.append(
                {
                    "image_id": image_id,
                    "image_path": image_path
                }
            )

            if index % 50 == 0 or index == total:

                print(
                    f"Processed {index}/{total}"
                )

        except Exception as e:

            print()
            print(
                f"ERROR: {image_path}"
            )

            print(e)

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    with open(
        IDS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "image_path"
            ]
        )

        writer.writeheader()

        writer.writerows(
            records
        )

    print()
    print("=" * 60)
    print("INDEX CREATED SUCCESSFULLY")
    print("=" * 60)

    print()
    print("Embeddings:", EMBEDDINGS_FILE)
    print("Image IDs:", IDS_FILE)
    print("Embedding shape:", embeddings.shape)
    print("Images indexed:", len(records))

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()