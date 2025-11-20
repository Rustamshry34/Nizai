from __future__ import annotations
from typing import List, Any, Dict, Tuple, Optional
import numpy as np
import asyncio


# ------------------------------------------------------------
# SIMPLE VECTOR STORE (FAISS-LIKE)
# ------------------------------------------------------------
class SimpleVectorStore:
    """
    Minimal in-memory vector store.
    Vectors stored as numpy arrays; metadata stored alongside.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vec: np.ndarray, meta: Dict[str, Any]):
        if vec.shape[0] != self.dim:
            raise ValueError(f"Vector dim mismatch: expected {self.dim}, got {vec.shape[0]}")
        self.vectors.append(vec)
        self.metadata.append(meta)

    def search_by_vector(self, query_vec: np.ndarray, k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
        if len(self.vectors) == 0:
            return []

        mat = np.stack(self.vectors)  # [N, D]
        # cosine similarity
        # add small eps to denominators to avoid NaN
        mat_norm = np.linalg.norm(mat, axis=1) + 1e-8
        q_norm = np.linalg.norm(query_vec) + 1e-8
        sims = (mat @ query_vec) / (mat_norm * q_norm)
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.metadata[i]) for i in idx]


# ------------------------------------------------------------
# EPISODIC MEMORY
# ------------------------------------------------------------
class EpisodicMemory:
    """
    Ordered agent experiences.
    """

    def __init__(self, max_items: int = 200):
        self.max_items = max_items
        self.memory: List[Dict[str, Any]] = []
        # simple lock for concurrent writes
        self._lock = asyncio.Lock()

    async def add(self, item: Dict[str, Any]):
        async with self._lock:
            self.memory.append(item)
            if len(self.memory) > self.max_items:
                self.memory.pop(0)

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self.memory)

    def search_last(self, k: int = 3) -> List[Dict[str, Any]]:
        return list(self.memory[-k:])


# ------------------------------------------------------------
# SEMANTIC RAG MEMORY
# ------------------------------------------------------------
class SemanticMemory:
    """
    Embedding-based memory. Uses provided llm_adapter.embed(texts) async method.
    """

    def __init__(self, llm_adapter, dim: int = 768):
        self.llm = llm_adapter
        self.index = SimpleVectorStore(dim=dim)

    async def add(self, text: str, metadata: Dict[str, Any] = None):
        if metadata is None:
            metadata = {}
        # LLM embed expects list[str] -> returns List[List[float]]
        vectors = await self.llm.embed([text])
        if not vectors or not isinstance(vectors[0], (list, tuple, np.ndarray)):
            raise RuntimeError("Embed returned unexpected format")
        vec = np.array(vectors[0], dtype=float)
        # if vector dim mismatch, SimpleVectorStore will raise
        self.index.add(vec, {"text": text, **metadata})

    async def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        vectors = await self.llm.embed([query])
        if not vectors:
            return []
        qvec = np.array(vectors[0], dtype=float)
        hits = self.index.search_by_vector(qvec, k=k)
        # return list of metadata with similarity score
        return [{"score": s, **meta} for s, meta in hits]


# ------------------------------------------------------------
# HYBRID MEMORY CONTROLLER
# ------------------------------------------------------------
class MemorySystem:
    """
    Hybrid memory controller:
      - episodic.add(...) / search_last(...)
      - semantic.add(...) / search(...)
      - retrieve_context(query) -> {semantic: [...], episodic: [...]}
    """

    def __init__(self, llm_adapter, episodic_limit: int = 200, embed_dim: int = 768):
        self.episodic = EpisodicMemory(max_items=episodic_limit)
        self.semantic = SemanticMemory(llm_adapter, dim=embed_dim)

    # Agent writes its steps (episodic)
    async def add_episode(self, action: str, result: Any):
        await self.episodic.add({"action": action, "result": result})

    # Agent stores long-term knowledge (semantic)
    async def add_knowledge(self, text: str, metadata: Dict[str, Any] = None):
        await self.semantic.add(text, metadata or {})

    # Hybrid retrieval: returns semantic hits (with scores) and last episodic events
    async def retrieve_context(self, query: str, k_semantic: int = 3, k_episodic: int = 3) -> Dict[str, Any]:
        # run semantic search and episodic search concurrently
        sem_task = asyncio.create_task(self.semantic.search(query, k=k_semantic))
        # episodic search is sync-ish, so just call
        epi = self.episodic.search_last(k=k_episodic)
        sem = await sem_task
        return {"semantic": sem, "episodic": epi}
    

    
