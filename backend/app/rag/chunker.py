import re
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ChildChunk(BaseModel):
    child_id: str
    parent_id: str
    doc_id: str
    text: str
    word_count: int
    char_start: int
    char_end: int
    language: str = "en"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ParentChunk(BaseModel):
    parent_id: str
    doc_id: str
    text: str
    word_count: int
    language: str = "en"
    children: List[ChildChunk] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HierarchicalChunker:
    """
    Advanced Multi-Tier Chunking Engine designed for MSMARCO-XI dataset.
    Implements:
    1. Document / Passage boundary detection
    2. Parent chunk segmentation (Broad semantic context, ~300-450 words)
    3. Child chunk extraction (High-precision retrieval units, ~70-120 words with overlap)
    4. Sentence boundary preservation
    5. Rich metadata tracking (doc_id, parent_id, language, source)
    """

    def __init__(
        self,
        parent_chunk_size: int = 350,
        parent_overlap: int = 40,
        child_chunk_size: int = 90,
        child_overlap: int = 25
    ):
        self.parent_chunk_size = parent_chunk_size
        self.parent_overlap = parent_overlap
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap

    def split_into_sentences(self, text: str) -> List[str]:
        """Splits text into sentences using regex boundary detection preserving punctuation."""
        if not text:
            return []
        # Support Latin sentence terminators (.!?) and Indic danda (।)
        sentence_endings = re.compile(r'(?<=[.!?।])\s+')
        sentences = [s.strip() for s in sentence_endings.split(text) if s.strip()]
        return sentences if sentences else [text]

    def _create_sliding_windows(self, words: List[str], window_size: int, overlap: int) -> List[tuple[str, int, int]]:
        """Creates sliding windows of words with given size and overlap."""
        step = max(1, window_size - overlap)
        chunks = []
        for i in range(0, len(words), step):
            window_words = words[i:i + window_size]
            if not window_words:
                break
            chunk_text = " ".join(window_words)
            chunks.append((chunk_text, i, i + len(window_words)))
            if i + window_size >= len(words):
                break
        return chunks

    def chunk_document(
        self,
        doc_id: str,
        text: str,
        language: str = "en",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[ParentChunk]:
        """
        Takes raw document text and produces hierarchical Parent-Child chunks.
        """
        meta = metadata or {}
        text = text.strip()
        if not text:
            return []

        words = text.split()
        parent_chunks: List[ParentChunk] = []

        # If document is short enough to be a single parent passage
        if len(words) <= self.parent_chunk_size:
            parent_id = f"{doc_id}_p0"
            parent = ParentChunk(
                parent_id=parent_id,
                doc_id=doc_id,
                text=text,
                word_count=len(words),
                language=language,
                metadata=meta
            )
            # Create child chunks
            child_windows = self._create_sliding_windows(words, self.child_chunk_size, self.child_overlap)
            for c_idx, (c_text, start_idx, end_idx) in enumerate(child_windows):
                child = ChildChunk(
                    child_id=f"{parent_id}_c{c_idx}",
                    parent_id=parent_id,
                    doc_id=doc_id,
                    text=c_text,
                    word_count=len(c_text.split()),
                    char_start=start_idx,
                    char_end=end_idx,
                    language=language,
                    metadata=meta
                )
                parent.children.append(child)
            parent_chunks.append(parent)
            return parent_chunks

        # Otherwise, segment into parent chunks with overlap
        parent_windows = self._create_sliding_windows(words, self.parent_chunk_size, self.parent_overlap)
        for p_idx, (p_text, p_start, p_end) in enumerate(parent_windows):
            parent_id = f"{doc_id}_p{p_idx}"
            p_words = p_text.split()
            parent = ParentChunk(
                parent_id=parent_id,
                doc_id=doc_id,
                text=p_text,
                word_count=len(p_words),
                language=language,
                metadata=meta
            )
            # Create child chunks for this parent window
            child_windows = self._create_sliding_windows(p_words, self.child_chunk_size, self.child_overlap)
            for c_idx, (c_text, c_start, c_end) in enumerate(child_windows):
                child = ChildChunk(
                    child_id=f"{parent_id}_c{c_idx}",
                    parent_id=parent_id,
                    doc_id=doc_id,
                    text=c_text,
                    word_count=len(c_text.split()),
                    char_start=c_start,
                    char_end=c_end,
                    language=language,
                    metadata=meta
                )
                parent.children.append(child)
            parent_chunks.append(parent)

        return parent_chunks
