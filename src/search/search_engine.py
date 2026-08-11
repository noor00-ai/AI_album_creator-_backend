import csv
import numpy as np

from embedding_engine import EmbeddingEngine


class SearchEngine:

    def __init__(self):

        print("=" * 60)
        print("INITIALIZING SEARCH ENGINE")
        print("=" * 60)

        self.embedding_engine = EmbeddingEngine()

        self.embeddings_path = "outputs/search/image_embeddings.npy"
        self.ids_path = "outputs/search/image_ids.csv"

        print("Loading image embeddings...")

        self.image_embeddings = np.load(self.embeddings_path)

        self.image_ids = []

        with open(
            self.ids_path,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:
                self.image_ids.append(row["image_id"])

        print(f"Embeddings loaded: {self.image_embeddings.shape}")
        print(f"Image IDs loaded: {len(self.image_ids)}")

        print("=" * 60)
        print("SEARCH ENGINE READY")
        print("=" * 60)

    def search(self, query, top_k=5):

        print()
        print(f"Searching for: {query}")

        query_embedding = self.embedding_engine.encode_text(query)

        similarities = self.image_embeddings @ query_embedding

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for index in top_indices:

            results.append({
                "image_id": self.image_ids[index],
                "similarity": float(similarities[index])
            })

        return results


if __name__ == "__main__":

    engine = SearchEngine()

    query = input("\nEnter your search query: ")

    results = engine.search(query, top_k=5)

    print()
    print("=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)

    for rank, result in enumerate(results, start=1):

        print(
            f"{rank}. "
            f"Image ID: {result['image_id']} | "
            f"Similarity: {result['similarity']:.4f}"
        )

    print("=" * 60)