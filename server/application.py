from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SearchParams, Distance, VectorParams
from fastapi.middleware.cors import CORSMiddleware

import os, uuid
import requests

load_dotenv()


#
##
### Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "demo")

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

app = FastAPI()

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


#
##
### Helper functions for querying Qdrant database
def does_collection_exist(protocol, hostname, port, collection_name):
    """Check if a specific collection exists by fetching all collections and searching for the name."""
    url = f"{protocol}://{hostname}:{port}/collections"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # Parse the collections list from the nested 'result' -> 'collections' structure
        existing_collection = [
            collection["name"]
            for collection in data.get("result", {}).get("collections", [])
        ]
        return collection_name in existing_collection
    except requests.RequestException as e:
        print(f"HTTP Request failed: {e}")
        return False


#
##
### API routes


@app.get("/debug")
async def debug():
    return {"env": dict(os.environ)}


class TextInput(BaseModel):
    text: str
    apiKey: str


@app.post("/api/generate-and-store-embeddings/")
async def generate_embeddings(input: TextInput):
    try:
        # https://platform.openai.com/account/api-keys
        # Generate embeddings using OpenAI's API
        client = OpenAI(api_key=input.apiKey)
        response = client.embeddings.create(
            model="text-embedding-3-small", input=input.text
        )
        embeddings = (
            response["data"][0]["embedding"]
            if isinstance(response, dict)
            else response.data[0].embedding
        )
        vector_id = str(uuid.uuid4())

        collection_name = COLLECTION_NAME
        collection_exists = does_collection_exist(
            "http", QDRANT_HOST, QDRANT_PORT, collection_name
        )

        if not collection_exists:
            try:
                qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=1536,
                        distance=Distance.COSINE,
                    ),
                )
                print(f"New collection {COLLECTION_NAME} is created successfully")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # Prepare a point for Qdrant insertion
        point = PointStruct(
            id=vector_id, vector=embeddings, payload={"text": input.text}
        )

        # Insert the point into Qdrant
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])
        return {"message": "Embeddings generated and stored successfully"}

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# @app.options("/api/retrieve-and-generate-response/")
# async def options_retrieve_and_generate_response():
#    return {"Allow": "POST, OPTIONS"}


@app.post("/api/retrieve-and-generate-response/")
async def retrieve_and_generate_response(input: TextInput):
    try:
        client = OpenAI(api_key=input.apiKey)
        response = client.embeddings.create(
            model="text-embedding-3-small", input=input.text
        )
        embeddings = (
            response["data"][0]["embedding"]
            if isinstance(response, dict)
            else response.data[0].embedding
        )

        # Search for the embeddings in Qdrant
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME, query_vector=embeddings, limit=3
        )

        documents = (
            [{"text": doc.payload["text"], "score": doc.score} for doc in search_result]
            if search_result
            else []
        )

        prompt = f"Question: {input.text}\n\n" + "\n\n".join(
            [f"Document {i+1}: {doc['text']}" for i, doc in enumerate(documents)]
        )

        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return {
            "response": completion.choices[0].message.content,
            "documents": documents,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
