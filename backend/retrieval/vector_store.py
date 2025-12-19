"""Vector store implementation using ChromaDB."""

from typing import List, Dict, Optional, Union, Sequence, Any
import chromadb
from chromadb.config import Settings
import numpy as np
import logging
from pathlib import Path
from uuid import uuid4

from backend.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


class VectorStore:
    """ChromaDB-based vector store for document chunks."""
    
    def _collection_metadata(self) -> Dict[str, Any]:
        """
        Build collection metadata for HNSW.
        
        Chroma's accepted HNSW metadata keys can vary by version. Some versions
        use:
          - hnsw:construction_ef
          - hnsw:search_ef
        while others may reject unknown keys.
        
        We'll attempt the richer set first and fall back if the server rejects it.
        """
        md: Dict[str, Any] = {"hnsw:space": "cosine"}
        # Best-effort: these keys are accepted by many Chroma versions
        md["hnsw:M"] = int(getattr(config, "HNSW_M", 32))
        md["hnsw:construction_ef"] = int(getattr(config, "HNSW_EF_CONSTRUCTION", 200))
        md["hnsw:search_ef"] = int(getattr(config, "HNSW_EF_SEARCH", 128))
        return md
    
    def __init__(
        self,
        collection_name: str = "audit_documents",
        persist_directory: str = None
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or config.VECTOR_DB_PATH
        
        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            try:
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata=self._collection_metadata(),
                )
            except Exception as e:
                # Fall back to minimal metadata if this Chroma build rejects HNSW params
                logger.warning("Failed to create collection with HNSW params; retrying with minimal metadata. Error: %s", e)
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            
            logger.info(f"Collection '{collection_name}' initialized with {self.collection.count()} documents")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(
        self,
        chunks: List[Dict],
        embeddings: Union[np.ndarray, Sequence[Any]],
        ids: Optional[List[str]] = None
    ) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
            embeddings: Array of embeddings for the chunks
            ids: Optional list of IDs for the chunks
            
        Returns:
            Number of documents added
        """
        if not chunks or len(embeddings) == 0:
            logger.warning("No chunks or embeddings provided")
            return 0
        
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        try:
            if ids is None:
                ids = [f"doc_{uuid4().hex}" for _ in range(len(chunks))]

            texts = [chunk['text'] for chunk in chunks]
            metadatas = [chunk.get('metadata', {}) for chunk in chunks]

            if isinstance(embeddings, np.ndarray):
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                embeddings_list: List[List[float]] = embeddings.astype(float).tolist()
            else:
                embeddings_seq = list(embeddings)
                if embeddings_seq and isinstance(embeddings_seq[0], np.ndarray):
                    embeddings_list = [e.astype(float).tolist() for e in embeddings_seq]  
                else:
                    embeddings_list = embeddings_seq  

            if hasattr(self.collection, "upsert"):
                self.collection.upsert(
                    documents=texts,
                    embeddings=embeddings_list,
                    metadatas=metadatas,
                    ids=ids
                )
            else:
                try:
                    self.collection.delete(ids=ids)
                except Exception:
                    pass
                self.collection.add(
                    documents=texts,
                    embeddings=embeddings_list,
                    metadatas=metadatas,
                    ids=ids
                )
            
            logger.info(f"Added {len(chunks)} documents to collection")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def query(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Query the vector store for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            Dictionary with 'documents', 'metadatas', 'distances', 'ids'
        """
        try:
            if isinstance(query_embedding, np.ndarray):
                if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
                    query_embedding = query_embedding[0]
                elif query_embedding.ndim != 1:
                    query_embedding = np.asarray(query_embedding).reshape(-1)
                query_embedding = query_embedding.astype(float).tolist()
            else:
                # Might come in as [[...]]; flatten a single leading dimension
                if query_embedding and isinstance(query_embedding[0], list) and len(query_embedding) == 1:
                    query_embedding = query_embedding[0]  # type: ignore[assignment]
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = {
                'documents': results['documents'][0] if results['documents'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'distances': results['distances'][0] if results['distances'] else [],
                'ids': results['ids'][0] if results['ids'] else []
            }
            
            logger.info(f"Query returned {len(formatted_results['documents'])} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            raise
    
    def search_by_text(
        self,
        query_text: str,
        embedder,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search using text query (will be embedded internally).
        
        Args:
            query_text: Text query
            embedder: EmbeddingGenerator instance
            top_k: Number of results
            
        Returns:
            List of result dictionaries
        """
        # Generate embedding for query (ensure 1D vector)
        query_embedding = embedder.encode(query_text)
        if isinstance(query_embedding, np.ndarray) and query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
            query_embedding = query_embedding[0]
        
        # Query vector store
        results = self.query(query_embedding, top_k=top_k, filter_metadata=filter_metadata)
        
        # Format as list of dicts
        formatted = []
        for i in range(len(results['documents'])):
            formatted.append({
                'text': results['documents'][i],
                'metadata': results['metadatas'][i],
                'distance': results['distances'][i],
                'id': results['ids'][i],
                'score': 1 - results['distances'][i]  # Convert distance to similarity
            })
        
        return formatted
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
    
    def reset_collection(self):
        """Reset (delete and recreate) the collection."""
        try:
            self.delete_collection()
            try:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata=self._collection_metadata(),
                )
            except Exception as e:
                logger.warning("Failed to recreate collection with HNSW params; retrying with minimal metadata. Error: %s", e)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            logger.info(f"Reset collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            raise
    
    def get_count(self) -> int:
        """Get number of documents in collection."""
        return self.collection.count()
    
    def get_all_documents(self) -> Dict:
        """Get all documents from the collection."""
        try:
            results = self.collection.get()
            return results
        except Exception as e:
            logger.error(f"Error getting all documents: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    from ingestion.embedder import EmbeddingGenerator
    
    # Initialize
    vector_store = VectorStore()
    embedder = EmbeddingGenerator()
    
    # Sample data
    chunks = [
        {'text': 'Quality management ensures compliance.', 'metadata': {'source': 'doc1'}},
        {'text': 'Verification is a critical process.', 'metadata': {'source': 'doc2'}},
    ]
    
    # Generate embeddings
    embeddings = embedder.encode([c['text'] for c in chunks])
    
    # Add to store
    vector_store.add_documents(chunks, embeddings)
    
    # Search
    results = vector_store.search_by_text("quality verification", embedder, top_k=2)
    
    print(f"Found {len(results)} results")
    for result in results:
        print(f"- {result['text'][:50]}... (score: {result['score']:.3f})")

