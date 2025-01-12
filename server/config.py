import os

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "demo")

OLAMA_URL = os.getenv("OLAMA_URL", "http://localhost:11434")
OLAMA_MODEL = os.getenv("OLAMA_MODEL", "llama3.2")
