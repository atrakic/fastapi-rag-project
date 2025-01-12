from functools import lru_cache
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from ollama import Client
#from ollama import chat
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from fastapi.middleware.cors import CORSMiddleware
from utils import collection_exist
from config import OLAMA_URL, OLAMA_MODEL, QDRANT_URL, COLLECTION_NAME
from models import TextInput
import os
import uuid

load_dotenv()

qdrant_client = QdrantClient(url=QDRANT_URL, port=6333)

app = FastAPI(
    description="API to interact with Qdrant and OLAMA",
)

app.add_middleware(
    CORSMiddleware,
    # Allows access from your React app
    allow_origins=[
        "http://client:3000",
        "http://localhost:3000",
        "https://salmon-moss-08b81541e.5.azurestaticapps.net",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

### API routes

@app.get("/api/debug/")
async def debug():
    return {"env": dict(os.environ)}

# https://ollama.com/blog/embedding-models
@app.post("/api/generate-and-store-embeddings/")
async def generate_embeddings(input: TextInput):
    try:
        oclient = Client(host=OLAMA_URL)

        # Generate embeddings
        response = oclient.embeddings(model=OLAMA_MODEL, prompt=input.text)
        embeddings = response["embedding"]

        # Create a collection if it doesn't already exist
        collection_exists = collection_exist(
            qdrnt_url=QDRANT_URL, collection_name=COLLECTION_NAME
        )
        if not collection_exists:
            try:
                qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=len(embeddings),
                        distance=Distance.COSINE,
                    ),
                )
                print(f"New collection {COLLECTION_NAME} is created successfully")
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to create collection: {str(e)}"
                )

        # Upload the vectors to the collection along with the original text as payload
        vector_id = str(uuid.uuid4())
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=vector_id, vector=embeddings, payload={"text": input.text}
                )
            ],
        )
        return {"message": "Embeddings generated and stored successfully"}

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate and store embeddings: {str(e)}")

@lru_cache
@app.post("/api/retrieve-and-generate-response/")
async def retrieve_and_generate_response(input: TextInput):
    try:
        oclient = Client(host=OLAMA_URL)

        # Generate embeddings
        response = oclient.embeddings(model=OLAMA_MODEL, prompt=input.text)
        embeddings = response["embedding"]

        # Search for similar vectors
        hits = qdrant_client.search(
            collection_name=COLLECTION_NAME, query_vector=embeddings, limit=3
        )

        documents = (
            [{"text": doc.payload["text"], "score": doc.score} for doc in hits]
            if hits
            else []
        )

        prompt = f"Question: {input.text}\n\n" + "\n\n".join(
            [f"Document {i+1}: {doc['text']}" for i, doc in enumerate(documents)]
        )

        completion = oclient.chat(
            model=OLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )

        return {
            "response": completion["message"]["content"],
            "documents": documents,
        }

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve and generate response: {str(e)}")
