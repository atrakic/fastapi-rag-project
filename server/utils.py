import requests


#
##
### Helper functions for querying Qdrant database
def collection_exist(qdrnt_url, collection_name):
    """Check if a specific collection exists by fetching all collections and searching for the name."""
    url = f"{qdrnt_url}/collections"
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
