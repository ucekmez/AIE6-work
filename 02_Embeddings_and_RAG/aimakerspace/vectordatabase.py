import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Any
from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.text_utils import Document # Import Document class
import asyncio
import uuid # For generating unique IDs

def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    if not isinstance(vector_a, np.ndarray) or not isinstance(vector_b, np.ndarray):
        return 0.0 # Or handle error appropriately
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0 # Avoid division by zero
    dot_product = np.dot(vector_a, vector_b)
    return dot_product / (norm_a * norm_b)

class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        # Store documents by a unique ID
        # Each entry contains: {"content": str, "vector": np.array, "metadata": Dict}
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, document: Document) -> str:
        """Inserts a Document object into the database."""
        # Generate embedding for the document content
        vector = self.embedding_model.get_embedding(document.page_content)
        # Generate a unique ID for the document
        doc_id = str(uuid.uuid4())
        self.documents[doc_id] = {
            "content": document.page_content,
            "vector": np.array(vector),
            "metadata": document.metadata
        }
        return doc_id

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
    ) -> List[Tuple[str, float, Dict[str, Any]]]: # Returns ID, score, metadata
        """Searches for the k nearest neighbors to a query vector."""
        scores = [
            (doc_id, distance_measure(query_vector, data["vector"]), data["metadata"])
            for doc_id, data in self.documents.items()
        ]
        # Sort by score in descending order
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
    ) -> List[Tuple[str, float, Dict[str, Any]]]: # Returns content, score, metadata
        """Searches for the k nearest neighbors to a query text."""
        query_vector = self.embedding_model.get_embedding(query_text)
        search_results = self.search(query_vector, k, distance_measure)
        # Return content, score, metadata for each result
        return [
            (self.documents[doc_id]["content"], score, metadata)
            for doc_id, score, metadata in search_results
        ]

    def retrieve_from_id(self, doc_id: str) -> Dict[str, Any]:
        """Retrieves a document by its unique ID."""
        return self.documents.get(doc_id, None)

    async def abuild_from_list(self, documents: List[Document]) -> "VectorDatabase":
        """Builds the database from a list of Document objects."""
        print(f"Starting to build vector database from {len(documents)} documents...")
        # Get embeddings for all document contents in batches (using existing EmbeddingModel method)
        contents = [doc.page_content for doc in documents]
        embeddings = await self.embedding_model.async_get_embeddings(contents)
        
        # Insert each document with its content, embedding, and metadata
        for doc, embedding in zip(documents, embeddings):
            doc_id = str(uuid.uuid4())
            self.documents[doc_id] = {
                "content": doc.page_content,
                "vector": np.array(embedding),
                "metadata": doc.metadata
            }
        print(f"Successfully built vector database with {len(self.documents)} entries.")
        return self


if __name__ == "__main__":
    # Example usage with Document objects
    list_of_docs = [
        Document("I like to eat broccoli and bananas.", {"source": "doc1", "topic": "food"}),
        Document("I ate a banana and spinach smoothie for breakfast.", {"source": "doc1", "topic": "food"}),
        Document("Chinchillas and kittens are cute.", {"source": "doc2", "topic": "animals"}),
        Document("My sister adopted a kitten yesterday.", {"source": "doc2", "topic": "animals"}),
        Document("Look at this cute hamster munching on a piece of broccoli.", {"source": "doc3", "topic": "animals_food"}),
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_docs))
    k = 2

    # Search results now include metadata
    search_results = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"\nClosest {k} result(s) (content, score, metadata):")
    for content, score, metadata in search_results:
        print(f"  - Content: \"{content}\", Score: {score:.4f}, Metadata: {metadata}")

    # Example of retrieving by ID (requires knowing an ID, let's grab one from the search)
    if search_results:
        # Find the original ID corresponding to the first result's content
        first_result_content = search_results[0][0]
        found_id = None
        for doc_id, data in vector_db.documents.items():
            if data['content'] == first_result_content:
                found_id = doc_id
                break
        if found_id:
            retrieved_doc = vector_db.retrieve_from_id(found_id)
            print(f"\nRetrieved document by ID ({found_id}):", retrieved_doc)
        else:
            print("\nCould not find ID for first search result content to test retrieval.")
