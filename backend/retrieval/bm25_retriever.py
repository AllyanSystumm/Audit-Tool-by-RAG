"""BM25 retriever for keyword-based search."""

from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 retriever for keyword-based document retrieval."""
    
    def __init__(self):
        """Initialize BM25 retriever."""
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
    
    def index_documents(self, documents: List[Dict], text_key: str = 'text'):
        """
        Index documents for BM25 search.
        
        Args:
            documents: List of document dictionaries
            text_key: Key in document dict containing text
        """
        if not documents:
            logger.warning("No documents provided for indexing")
            return
        
        try:
            self.documents = documents
            
            # Tokenize documents (simple whitespace tokenization)
            self.tokenized_docs = [
                self._tokenize(doc[text_key])
                for doc in documents
            ]
            
            # Create BM25 index
            self.bm25 = BM25Okapi(self.tokenized_docs)
            
            logger.info(f"Indexed {len(documents)} documents for BM25 search")
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for documents using BM25.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of top matching documents with scores
        """
        if self.bm25 is None:
            logger.warning("BM25 index not initialized. Call index_documents first.")
            return []
        
        try:
            # Tokenize query
            tokenized_query = self._tokenize(query)
            
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            
            def _meta_matches(doc: Dict) -> bool:
                if not filter_metadata:
                    return True
                md = doc.get("metadata") or {}
                for k, v in filter_metadata.items():
                    if md.get(k) != v:
                        return False
                return True

            # Rank indices by score, then filter by metadata (best-effort; BM25 doesn't support per-query filtering natively)
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            top_indices = []
            for idx in ranked_indices:
                if _meta_matches(self.documents[idx]):
                    top_indices.append(idx)
                if len(top_indices) >= top_k:
                    break
            
            # Format results
            results = []
            for idx in top_indices:
                result = {
                    **self.documents[idx],
                    'bm25_score': float(scores[idx]),
                    'rank': len(results)
                }
                results.append(result)
            
            logger.info(f"BM25 search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error during BM25 search: {str(e)}")
            raise
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple tokenization: lowercase and split on whitespace
        # For better results, consider using nltk or spacy
        return text.lower().split()
    
    def get_document_count(self) -> int:
        """Get number of indexed documents."""
        return len(self.documents)


class AdvancedBM25Retriever(BM25Retriever):
    """
    Enhanced BM25 retriever with better tokenization.
    Requires nltk for stemming and stopword removal.
    """
    
    def __init__(self, use_stemming: bool = False, remove_stopwords: bool = True):
        """
        Initialize advanced BM25 retriever.
        
        Args:
            use_stemming: Whether to use stemming
            remove_stopwords: Whether to remove stopwords
        """
        super().__init__()
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        
        # Try to import nltk components
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import PorterStemmer
            
            self.stopwords = set(stopwords.words('english')) if remove_stopwords else set()
            self.stemmer = PorterStemmer() if use_stemming else None
            
        except ImportError:
            logger.warning("NLTK not available. Using simple tokenization.")
            self.stopwords = set()
            self.stemmer = None
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Advanced tokenization with optional stemming and stopword removal.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of processed tokens
        """
        # Lowercase and split
        tokens = text.lower().split()
        
        # Remove stopwords
        if self.remove_stopwords and self.stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        # Apply stemming
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        return tokens


# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        {
            'text': 'Quality management is essential for organizational success.',
            'metadata': {'doc_id': 1}
        },
        {
            'text': 'Verification processes ensure compliance with standards.',
            'metadata': {'doc_id': 2}
        },
        {
            'text': 'Documentation and quality control are critical.',
            'metadata': {'doc_id': 3}
        }
    ]
    
    # Initialize and index
    retriever = BM25Retriever()
    retriever.index_documents(documents)
    
    # Search
    query = "quality verification standards"
    results = retriever.search(query, top_k=2)
    
    print(f"Top results for '{query}':")
    for result in results:
        print(f"- {result['text'][:50]}... (score: {result['bm25_score']:.3f})")

