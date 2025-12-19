"""Semantic text chunking for document processing."""

from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import re

from backend.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticChunker:
    """Chunk text into semantically meaningful segments."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            separators: List of separators for splitting
        """
        self.chunk_size = chunk_size or config.MAX_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        
        # Hierarchical separators - try to break on meaningful boundaries
        self.separators = separators or [
            "\n\n\n",  # Multiple line breaks
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ".",       # Sentences
            "!",
            "?",
            ";",
            ":",
            ",",
            " ",       # Words
            ""         # Characters (last resort)
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text into semantically coherent segments.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of dictionaries with 'text' and 'metadata' for each chunk
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        try:
            # Heuristic: process/procedure documents often have headings.
            # If we detect sufficient structure, prefer section-aware chunking first.
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

            def _is_heading(line: str) -> bool:
                if not line or len(line) > 120:
                    return False
                if line.isupper() and len(line) >= 4 and len(line.split()) <= 8:
                    return True
                if line.endswith(":") and len(line.split()) <= 10:
                    return True
                if re.match(r"^\d+(\.\d+)*\s+.+", line):
                    return True
                if re.match(r"^(phase|section)\s+\d+", line, flags=re.IGNORECASE):
                    return True
                return False

            heading_count = sum(1 for ln in lines[:400] if _is_heading(ln))
            use_structured = heading_count >= 6

            if use_structured:
                structured = StructuredChunker(chunk_size=self.chunk_size).chunk_by_sections(text, metadata)
                chunk_dicts = [
                    {"text": c.get("text", ""), "metadata": c.get("metadata", {})}
                    for c in structured
                    if c.get("text")
                ]
            else:
                chunk_dicts = [{"text": t, "metadata": (metadata or {})} for t in self.splitter.split_text(text)]
            
            # Create chunk objects with metadata
            chunk_objects = []
            for i, cd in enumerate(chunk_dicts):
                chunk = cd.get("text", "")
                extra_md = cd.get("metadata", {}) or {}
                chunk_metadata = {
                    'chunk_id': i,
                    'total_chunks': len(chunk_dicts),
                    'char_count': len(chunk),
                    **extra_md
                }
                
                chunk_objects.append({
                    'text': chunk,
                    'metadata': chunk_metadata
                })
            
            logger.info(f"Created {len(chunk_objects)} chunks from text of length {len(text)}")
            return chunk_objects
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of documents with 'text' and 'metadata'
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            text = doc.get('text', '')
            doc_metadata = doc.get('metadata', {})
            doc_metadata['doc_id'] = doc_idx
            
            chunks = self.chunk_text(text, doc_metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} total chunks")
        return all_chunks


class StructuredChunker:
    """
    Chunker that preserves document structure (headings, sections).
    Better for structured audit documents.
    """
    
    def __init__(self, chunk_size: int = None):
        """Initialize structured chunker."""
        self.chunk_size = chunk_size or config.MAX_CHUNK_SIZE
    
    def chunk_by_sections(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text by detecting sections/headings.
        
        This is a simple implementation. For better results, consider using
        LangChain's MarkdownHeaderTextSplitter for markdown-formatted documents.
        """
        chunks = []
        
        # Split by common section markers
        sections = []
        current_section = []
        
        for line in text.split('\n'):
            # Detect section headers (simple heuristic)
            if self._is_section_header(line):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        # Add last section
        if current_section:
            sections.append('\n'.join(current_section))
        
        # Create chunks from sections
        for i, section in enumerate(sections):
            if len(section) > self.chunk_size:
                # Section too large, split further
                subsplitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=100
                )
                subsections = subsplitter.split_text(section)
                for j, subsection in enumerate(subsections):
                    chunks.append({
                        'text': subsection,
                        'metadata': {
                            'section_id': i,
                            'subsection_id': j,
                            **(metadata or {})
                        }
                    })
            else:
                chunks.append({
                    'text': section,
                    'metadata': {
                        'section_id': i,
                        **(metadata or {})
                    }
                })
        
        return chunks
    
    def _is_section_header(self, line: str) -> bool:
        """Detect if a line is a section header."""
        line = line.strip()
        
        # Common patterns for headers
        patterns = [
            line.isupper() and len(line) > 3,  # ALL CAPS
            line.endswith(':') and len(line.split()) < 8,  # Ends with colon
            any(line.startswith(prefix) for prefix in ['#', '##', '###']),  # Markdown
            line.split('.')[0].isdigit() if '.' in line else False,  # Numbered (1., 2., etc.)
        ]
        
        return any(patterns)


# Example usage
if __name__ == "__main__":
    chunker = SemanticChunker()
    
    sample_text = """
    This is a sample document about quality processes.
    
    Section 1: Introduction
    Quality management is essential for organizational success.
    
    Section 2: Verification Process
    The verification process involves multiple checkpoints.
    """
    
    chunks = chunker.chunk_text(sample_text)
    print(f"Created {len(chunks)} chunks")
    for chunk in chunks:
        print(f"\nChunk {chunk['metadata']['chunk_id']}:")
        print(chunk['text'][:100])

