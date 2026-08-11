from embedding_engine import EmbeddingEngine


print("=" * 60)
print("EMBEDDING ENGINE TEST")
print("=" * 60)

engine = EmbeddingEngine()


image_path = "data/raw/images/ff/21/8985196546.jpg"

image_embedding = engine.encode_image(
    image_path
)

print()
print("IMAGE TEST")
print("Embedding shape:", image_embedding.shape)


query = "a person outdoors"

text_embedding = engine.encode_text(
    query
)

print()
print("TEXT TEST")
print("Query:", query)
print("Embedding shape:", text_embedding.shape)


similarity = float(
    image_embedding @ text_embedding
)

print()
print("SIMILARITY TEST")
print("Similarity:", similarity)


print()
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)