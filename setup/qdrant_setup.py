from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client import QdrantClient
import os


def main():
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=512,  # 4,
                distance=Distance.COSINE,
            ),
        )
        print(f"Collection {COLLECTION_NAME} created successfully")


def cities_points(client):
    operation_info = client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=[
            PointStruct(
                id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}
            ),
            PointStruct(
                id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}
            ),
            PointStruct(
                id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}
            ),
            PointStruct(
                id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}
            ),
            PointStruct(
                id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}
            ),
            PointStruct(
                id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}
            ),
        ],
    )
    print(operation_info)


if __name__ == "__main__":
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "demo")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    main()
    # cities_points(client)
