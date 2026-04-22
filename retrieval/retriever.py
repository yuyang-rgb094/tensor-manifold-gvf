"""
Unified Retriever for Tensor Manifold GVF.

Implements the core retrieval pipeline using modular components:
  1. EmbeddingEncoder  → encode raw text into dense vectors
  2. SignatureBuilder  → construct tensor signatures from entity-relation triples
  3. ManifoldProjector → project signatures onto tensor manifold space
  4. VectorIndex       → index manifold embeddings for efficient search
  5. TensorDecomposer  → CP/Tucker decomposition for multi-aspect retrieval

References:
  - Algorithm 1: Tensor Signature Construction
  - Algorithm 2: Grassmann Vector Field Retrieval
  - Algorithm 3: Incremental Manifold Update
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .encoders import create_encoder, EmbeddingEncoder
from .index import create_index, VectorIndex
from .pipeline import ManifoldProjector, SignatureBuilder, TensorDecomposer, DecompositionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Single retrieval result with score and optional decomposition info."""

    id: str
    title: str
    abstract: str
    score: float
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    decomposition: Optional[Dict[str, Any]] = None
    aspect_scores: Optional[Dict[str, float]] = None
    related_nodes: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Unified Retriever (Pipeline Orchestrator)
# ---------------------------------------------------------------------------

class UnifiedRetriever:
    """
    Unified retriever combining tensor manifold embeddings with
    Grassmann vector field search and CP decomposition.

    This class is a **pure pipeline orchestrator** — all concrete logic
    is delegated to pluggable components:

    * :class:`~.encoders.base.EmbeddingEncoder` — text → dense vectors
    * :class:`~.pipeline.signature_builder.SignatureBuilder` — docs → tensor signatures
    * :class:`~.pipeline.manifold_projector.ManifoldProjector` — signatures → manifold embeddings
    * :class:`~.index.base.VectorIndex` — manifold embeddings → searchable index
    * :class:`~.pipeline.decomposer.TensorDecomposer` — node → multi-aspect analysis

    Pipeline
    --------
    build():
        encode  →  signatures  →  manifold project  →  index  →  decomposer

    search(query_text):
        encode query  →  manifold project  →  index search  →  rank  →  results
    """

    DEFAULT_SBERT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        sbert_model: Optional[str] = None,
        embedding_dim: int = 384,
        manifold_dim: int = 64,
        index_type: str = "faiss",
        decomposer_type: str = "cp",
        rank: int = 8,
        manifold_mode: str = "truncate",
    ):
        """
        Parameters
        ----------
        config : dict, optional
            Full configuration dictionary (overrides individual args).
        sbert_model : str, optional
            Sentence-BERT model name or path.
        embedding_dim : int
            Dimensionality of SBERT embeddings.
        manifold_dim : int
            Dimensionality of manifold embeddings.
        index_type : str
            Backend index type: ``"faiss"``, ``"hnswlib"``, or ``"brute"``.
        decomposer_type : str
            Tensor decomposition method: ``"cp"`` or ``"tucker"``.
        rank : int
            Rank for CP / Tucker decomposition.
        manifold_mode : str
            Manifold projection mode: ``"truncate"`` or ``"learned"``.
        """
        if config is not None:
            sbert_model = config.get("sbert_model", sbert_model)
            embedding_dim = config.get("embedding_dim", embedding_dim)
            manifold_dim = config.get("manifold_dim", manifold_dim)
            index_type = config.get("index_type", index_type)
            decomposer_type = config.get("decomposer_type", decomposer_type)
            rank = config.get("rank", rank)
            manifold_mode = config.get("manifold_mode", manifold_mode)

        self.sbert_model = sbert_model or self.DEFAULT_SBERT_MODEL
        self.embedding_dim = embedding_dim
        self.manifold_dim = manifold_dim
        self.index_type = index_type
        self.decomposer_type = decomposer_type
        self.rank = rank
        self.manifold_mode = manifold_mode

        # ------------------------------------------------------------------
        # Pipeline components (populated by build / _init_components)
        # ------------------------------------------------------------------
        self._encoder: Optional[EmbeddingEncoder] = None
        self._signature_builder: Optional[SignatureBuilder] = None
        self._projector: Optional[ManifoldProjector] = None
        self._index: Optional[VectorIndex] = None
        self._decomposer: Optional[TensorDecomposer] = None

        # Internal state
        self._documents: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None
        self._manifold_embeddings: Optional[np.ndarray] = None
        self._tensor_signatures: List[Dict[str, Any]] = []
        self._id_to_idx: Dict[str, int] = {}
        self._relation_types: List[str] = []
        self._built: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        documents: Sequence[Dict[str, Any]],
        relations: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> "UnifiedRetriever":
        """
        Build the full retrieval pipeline.

        Steps
        -----
        1. Encode documents with EmbeddingEncoder
        2. Construct tensor signatures via SignatureBuilder
        3. Project onto manifold via ManifoldProjector
        4. Build vector index
        5. Initialize tensor decomposer

        Parameters
        ----------
        documents : sequence of dict
            Each dict must have at least ``id``, ``title``, ``abstract``.
        relations : sequence of dict, optional
            Each dict: ``{"source": str, "target": str, "type": str}``.

        Returns
        -------
        self
        """
        relations = relations or []
        t0 = time.time()

        self._init_components()

        logger.info("Step 1/5: Encoding %d documents ...", len(documents))
        self._documents = list(documents)
        embeddings = self._encoder.encode_documents(self._documents)
        self._embeddings = embeddings

        logger.info("Step 2/5: Building tensor signatures ...")
        self._relation_types = self._signature_builder.get_relation_types(relations)
        signatures = self._signature_builder.build(self._documents, relations)
        self._tensor_signatures = signatures

        logger.info("Step 3/5: Manifold projection (mode=%s) ...", self.manifold_mode)
        manifold_emb = self._projector.project(embeddings, signatures)
        self._manifold_embeddings = manifold_emb

        logger.info("Step 4/5: Building %s index ...", self.index_type)
        self._index.build(manifold_emb)

        logger.info("Step 5/5: Initializing %s decomposer (rank=%d) ...",
                     self.decomposer_type, self.rank)
        self._decomposer.init(signatures, manifold_emb, self._relation_types)

        self._id_to_idx = {
            doc["id"]: idx for idx, doc in enumerate(self._documents)
        }
        self._built = True
        elapsed = time.time() - t0
        logger.info("Build complete in %.2f s  (%d docs, dim=%d)",
                     elapsed, len(documents), self.manifold_dim)
        return self

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_year: Optional[Tuple[int, int]] = None,
        filter_venue: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Text-based retrieval via manifold search.

        Parameters
        ----------
        query : str
            Natural-language query text.
        top_k : int
            Number of results to return.
        filter_year : (int, int), optional
            Year range filter ``[min_year, max_year]``.
        filter_venue : str, optional
            Venue name filter (case-insensitive substring match).

        Returns
        -------
        list[RetrievalResult]
        """
        self._ensure_built()

        # Encode and project query
        query_emb = self._encoder.encode_single(query)
        query_manifold = self._projector.project_single(query_emb)

        # Search index
        raw_scores, indices = self._index.search(query_manifold, top_k * 3)

        # Apply filters and rank
        results: List[RetrievalResult] = []
        for rank, (idx, score) in enumerate(zip(indices, raw_scores)):
            if idx < 0 or idx >= len(self._documents):
                continue
            doc = self._documents[idx]

            if filter_year is not None:
                doc_year = doc.get("year")
                if doc_year is None or not (
                    filter_year[0] <= doc_year <= filter_year[1]
                ):
                    continue
            if filter_venue is not None:
                doc_venue = doc.get("venue", "")
                if filter_venue.lower() not in doc_venue.lower():
                    continue

            results.append(
                RetrievalResult(
                    id=doc["id"],
                    title=doc.get("title", ""),
                    abstract=doc.get("abstract", ""),
                    score=float(score),
                    rank=len(results) + 1,
                    metadata={
                        "year": doc.get("year"),
                        "venue": doc.get("venue"),
                        "authors": doc.get("authors", []),
                        "keywords": doc.get("keywords", []),
                    },
                )
            )
            if len(results) >= top_k:
                break

        return results

    def search_with_decomposition(
        self,
        node_id: str,
        top_k: int = 10,
        aspects: Optional[List[str]] = None,
    ) -> Tuple[List[RetrievalResult], Optional[DecompositionResult]]:
        """
        Node-based retrieval with full tensor decomposition.

        Parameters
        ----------
        node_id : str
            Target node (document) identifier.
        top_k : int
            Number of related nodes to return.
        aspects : list[str], optional
            Specific aspects to analyse.

        Returns
        -------
        results : list[RetrievalResult]
        decomp : DecompositionResult or None
        """
        self._ensure_built()

        if node_id not in self._id_to_idx:
            logger.warning("Node '%s' not found in index.", node_id)
            return [], None

        idx = self._id_to_idx[node_id]
        node_manifold = self._manifold_embeddings[idx]

        # Search for neighbours
        raw_scores, indices = self._index.search(node_manifold, top_k * 3)

        # Decompose the node's tensor signature
        decomp = self._decomposer.decompose_node(idx, aspects)

        # Re-rank using aspect-weighted scores
        results: List[RetrievalResult] = []
        for rank, (nidx, score) in enumerate(zip(indices, raw_scores)):
            if nidx < 0 or nidx >= len(self._documents) or nidx == idx:
                continue
            doc = self._documents[nidx]

            if decomp is not None and decomp.aspect_contributions:
                weighted_score = score
                for aspect, weight in decomp.aspect_contributions.items():
                    if aspect in doc.get("keywords", []):
                        weighted_score *= (1.0 + 0.1 * weight)
                score = weighted_score

            results.append(
                RetrievalResult(
                    id=doc["id"],
                    title=doc.get("title", ""),
                    abstract=doc.get("abstract", ""),
                    score=float(score),
                    rank=len(results) + 1,
                    metadata={
                        "year": doc.get("year"),
                        "venue": doc.get("venue"),
                        "authors": doc.get("authors", []),
                        "keywords": doc.get("keywords", []),
                    },
                    decomposition=decomp.__dict__ if decomp else None,
                    aspect_scores=decomp.aspect_contributions if decomp else None,
                    related_nodes=self._get_related_nodes(doc["id"]),
                )
            )
            if len(results) >= top_k:
                break

        return results, decomp

    def incremental_update(
        self,
        new_documents: Sequence[Dict[str, Any]],
        new_relations: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Incrementally update the retriever with new documents (Algorithm 3).

        Parameters
        ----------
        new_documents : sequence of dict
            New documents to add.
        new_relations : sequence of dict, optional
            New relation triples.

        Returns
        -------
        dict
            Update statistics.
        """
        self._ensure_built()
        new_relations = new_relations or []
        t0 = time.time()

        n_before = len(self._documents)

        # Step 1: Encode new documents
        logger.info("Incremental: encoding %d new documents ...", len(new_documents))
        new_embeddings = self._encoder.encode_documents(list(new_documents))

        # Step 2: Build tensor signatures for new entries
        all_relations = self._collect_relations() + list(new_relations)
        new_signatures = self._signature_builder.build(list(new_documents), new_relations)

        # Step 3: Manifold encode new signatures
        new_manifold = self._projector.project(new_embeddings, new_signatures)

        # Step 4: Update manifold (Grassmannian mean shift)
        self._projector.update(new_manifold)

        # Step 5: Extend index
        self._index.add(new_manifold)

        # Merge state
        self._documents.extend(new_documents)
        self._embeddings = np.vstack([self._embeddings, new_embeddings])
        self._manifold_embeddings = np.vstack(
            [self._manifold_embeddings, new_manifold]
        )
        self._tensor_signatures.extend(new_signatures)
        self._relation_types = self._signature_builder.get_relation_types(all_relations)

        # Update decomposer state
        self._decomposer.init(self._tensor_signatures, self._manifold_embeddings, self._relation_types)

        for i, doc in enumerate(new_documents, start=n_before):
            self._id_to_idx[doc["id"]] = i

        elapsed = time.time() - t0
        n_added = len(new_documents)
        stats = {
            "n_added": n_added,
            "total": len(self._documents),
            "update_time_s": round(elapsed, 4),
        }
        logger.info("Incremental update done: %s", stats)
        return stats

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialize the retriever state to JSON."""
        state = {
            "config": {
                "sbert_model": self.sbert_model,
                "embedding_dim": self.embedding_dim,
                "manifold_dim": self.manifold_dim,
                "index_type": self.index_type,
                "decomposer_type": self.decomposer_type,
                "rank": self.rank,
                "manifold_mode": self.manifold_mode,
            },
            "documents": self._documents,
            "relation_types": self._relation_types,
            "id_to_idx": self._id_to_idx,
            "manifold_embeddings": (
                self._manifold_embeddings.tolist()
                if self._manifold_embeddings is not None
                else None
            ),
            "embeddings": (
                self._embeddings.tolist()
                if self._embeddings is not None
                else None
            ),
            "built": self._built,
        }
        json_str = json.dumps(state, ensure_ascii=False, indent=2)
        if path is not None:
            Path(path).write_text(json_str, encoding="utf-8")
            logger.info("Retriever state saved to %s", path)
        return json_str

    @classmethod
    def from_json(cls, path: str) -> "UnifiedRetriever":
        """Deserialize a retriever from a JSON file."""
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        config = state["config"]
        retriever = cls(config=config)
        retriever._documents = state["documents"]
        retriever._relation_types = state.get("relation_types", [])
        retriever._id_to_idx = state.get("id_to_idx", {})
        retriever._built = state.get("built", True)

        if state.get("manifold_embeddings") is not None:
            retriever._manifold_embeddings = np.array(state["manifold_embeddings"])
        if state.get("embeddings") is not None:
            retriever._embeddings = np.array(state["embeddings"])

        # Rebuild index from loaded embeddings
        if retriever._manifold_embeddings is not None:
            retriever._init_components()
            retriever._index.build(retriever._manifold_embeddings)
            # Reconstruct minimal signatures for decomposer
            retriever._tensor_signatures = [
                {"doc_id": doc["id"], "entities": [], "relations": [],
                 "shape": (0, 0, retriever.embedding_dim), "slices": [],
                 "entity_count": 0, "relation_count": len(retriever._relation_types) or 1}
                for doc in retriever._documents
            ]
            retriever._decomposer.init(
                retriever._tensor_signatures, retriever._manifold_embeddings,
                retriever._relation_types
            )

        logger.info("Retriever loaded from %s (%d docs)", path, len(retriever._documents))
        return retriever

    # ------------------------------------------------------------------
    # Internal: Component initialization
    # ------------------------------------------------------------------

    def _init_components(self) -> None:
        """Initialize all pipeline components."""
        # Encoder
        if self._encoder is None:
            encoder_config = {
                "type": "sentence_transformer",
                "model_name": self.sbert_model,
                "batch_size": 128,
                "max_length": 512,
                "use_cache": True,
            }
            self._encoder = create_encoder(encoder_config)

        # Signature builder
        if self._signature_builder is None:
            self._signature_builder = SignatureBuilder(
                embedding_dim=self.embedding_dim
            )

        # Manifold projector
        if self._projector is None:
            self._projector = ManifoldProjector(
                mode=self.manifold_mode,
                semantic_dim=self.embedding_dim,
                output_dim=self.manifold_dim,
            )

        # Vector index
        if self._index is None:
            self._index = create_index(self.index_type)

        # Tensor decomposer
        if self._decomposer is None:
            self._decomposer = TensorDecomposer(
                decomposer_type=self.decomposer_type,
                rank=self.rank,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_relations(self) -> List[Dict[str, Any]]:
        """Collect relations from existing tensor signatures."""
        all_rels = []
        for sig in self._tensor_signatures:
            for rel in sig.get("relations", []):
                all_rels.append(rel)
        return all_rels

    def _get_related_nodes(
        self, doc_id: str, max_nodes: int = 5
    ) -> List[Dict[str, Any]]:
        """Get related nodes for a document from its tensor signature."""
        idx = self._id_to_idx.get(doc_id)
        if idx is None:
            return []
        sig = self._tensor_signatures[idx]
        related = []
        for s in sig.get("slices", []):
            for target_id in s.get("targets", [])[:max_nodes]:
                if target_id in self._id_to_idx:
                    tidx = self._id_to_idx[target_id]
                    tdoc = self._documents[tidx]
                    related.append({
                        "id": target_id,
                        "title": tdoc.get("title", ""),
                        "relation_type": s.get("relation_type", "related"),
                    })
        return related[:max_nodes]

    def _ensure_built(self) -> None:
        """Raise if the retriever has not been built yet."""
        if not self._built:
            raise RuntimeError(
                "Retriever has not been built. Call build() first."
            )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"UnifiedRetriever("
            f"model={self.sbert_model!r}, "
            f"dim={self.manifold_dim}, "
            f"index={self.index_type!r}, "
            f"decomp={self.decomposer_type!r}, "
            f"manifold={self.manifold_mode!r}, "
            f"docs={len(self._documents)}, "
            f"built={self._built})"
        )

    def __len__(self) -> int:
        return len(self._documents)
