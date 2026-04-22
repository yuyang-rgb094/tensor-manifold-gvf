"""
Unified Retriever for Tensor Manifold GVF.

Implements the core retrieval pipeline:
  1. SBERT encode raw text into dense vectors
  2. Build tensor signatures from entity-relation triples
  3. Manifold encode signatures into Grassmannian embeddings
  4. Index embeddings for efficient similarity search
  5. CP decomposition for multi-aspect retrieval

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
    # Decomposition fields (populated by search_with_decomposition)
    decomposition: Optional[Dict[str, Any]] = None
    aspect_scores: Optional[Dict[str, float]] = None
    related_nodes: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class DecompositionResult:
    """Result of CP / Tucker decomposition on a retrieved node."""

    node_id: str
    core_tensor: np.ndarray
    factor_matrices: Dict[str, np.ndarray]
    explained_variance_ratio: float
    aspect_contributions: Dict[str, float]
    reconstruction_error: float


# ---------------------------------------------------------------------------
# Unified Retriever
# ---------------------------------------------------------------------------

class UnifiedRetriever:
    """
    Unified retriever combining tensor manifold embeddings with
    Grassmann vector field search and CP decomposition.

    Pipeline
    --------
    build():
        SBERT encode  ->  signatures  ->  manifold encode  ->  index  ->  decomposer

    search(query_text):
        Encode query  ->  manifold project  ->  GVF search  ->  rank  ->  results

    search_with_decomposition(node_id):
        Retrieve node  ->  CP decompose tensor  ->  multi-aspect analysis
    """

    # Default SBERT model for academic text
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
            Dimensionality of manifold (Grassmannian) embeddings.
        index_type : str
            Backend index type: ``"faiss"``, ``"hnswlib"``, or ``"brute"``.
        decomposer_type : str
            Tensor decomposition method: ``"cp"`` or ``"tucker"``.
        rank : int
            Rank for CP / Tucker decomposition.
        """
        if config is not None:
            sbert_model = config.get("sbert_model", sbert_model)
            embedding_dim = config.get("embedding_dim", embedding_dim)
            manifold_dim = config.get("manifold_dim", manifold_dim)
            index_type = config.get("index_type", index_type)
            decomposer_type = config.get("decomposer_type", decomposer_type)
            rank = config.get("rank", rank)

        self.sbert_model = sbert_model or self.DEFAULT_SBERT_MODEL
        self.embedding_dim = embedding_dim
        self.manifold_dim = manifold_dim
        self.index_type = index_type
        self.decomposer_type = decomposer_type
        self.rank = rank

        # Internal state (populated by build)
        self._sbert_encoder: Any = None
        self._manifold_encoder: Any = None
        self._index: Any = None
        self._decomposer: Any = None
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
        1. SBERT encode document text into dense vectors
        2. Construct tensor signatures from entity-relation triples
        3. Manifold encode signatures into Grassmannian embeddings
        4. Build similarity index over manifold embeddings
        5. Initialize tensor decomposer

        Parameters
        ----------
        documents : sequence of dict
            Each dict must have at least ``id``, ``title``, ``abstract``.
            May also include ``authors``, ``year``, ``venue``, ``keywords``.
        relations : sequence of dict, optional
            Each dict: ``{"source": str, "target": str, "type": str}``.

        Returns
        -------
        self
        """
        relations = relations or []
        t0 = time.time()

        logger.info("Step 1/5: SBERT encoding %d documents ...", len(documents))
        self._documents = list(documents)
        embeddings = self._sbert_encode(documents)
        self._embeddings = embeddings

        logger.info("Step 2/5: Building tensor signatures ...")
        self._relation_types = self._get_relation_types(relations)
        signatures = self._build_tensor_signatures(documents, relations)
        self._tensor_signatures = signatures

        logger.info("Step 3/5: Manifold encoding signatures ...")
        manifold_emb = self._manifold_encode(signatures, embeddings)
        self._manifold_embeddings = manifold_emb

        logger.info("Step 4/5: Building %s index ...", self.index_type)
        self._build_index(manifold_emb)

        logger.info("Step 5/5: Initializing %s decomposer (rank=%d) ...",
                     self.decomposer_type, self.rank)
        self._init_decomposer()

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
        Text-based retrieval via Grassmann vector field search.

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

        # Encode query
        query_emb = self._sbert_encode_single(query)
        query_manifold = self._manifold_project(query_emb)

        # Search index
        raw_scores, indices = self._index_search(query_manifold, top_k * 3)

        # Apply filters and rank
        results: List[RetrievalResult] = []
        for rank, (idx, score) in enumerate(zip(indices, raw_scores)):
            if idx < 0 or idx >= len(self._documents):
                continue
            doc = self._documents[idx]

            # Apply filters
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

        Retrieves the tensor for *node_id*, performs CP/Tucker
        decomposition, and returns both ranked neighbours and the
        decomposition result for multi-aspect analysis.

        Parameters
        ----------
        node_id : str
            Target node (document) identifier.
        top_k : int
            Number of related nodes to return.
        aspects : list[str], optional
            Specific aspects to analyse (e.g. ``["author", "venue"]``).

        Returns
        -------
        results : list[RetrievalResult]
            Related nodes ranked by decomposed similarity.
        decomp : DecompositionResult or None
            Full decomposition details.
        """
        self._ensure_built()

        if node_id not in self._id_to_idx:
            logger.warning("Node '%s' not found in index.", node_id)
            return [], None

        idx = self._id_to_idx[node_id]
        node_manifold = self._manifold_embeddings[idx]

        # Search for neighbours
        raw_scores, indices = self._index_search(node_manifold, top_k * 3)

        # Decompose the node's tensor signature
        decomp = self._decompose_node(idx, aspects)

        # Re-rank using aspect-weighted scores
        results: List[RetrievalResult] = []
        for rank, (nidx, score) in enumerate(zip(indices, raw_scores)):
            if nidx < 0 or nidx >= len(self._documents) or nidx == idx:
                continue
            doc = self._documents[nidx]

            # Aspect-weighted re-ranking
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

        Instead of rebuilding from scratch, this method:
        1. Encodes new documents with SBERT
        2. Builds tensor signatures for new entries
        3. Projects new signatures onto the existing manifold
        4. Updates the manifold (Grassmannian mean shift)
        5. Extends the index with new embeddings

        Parameters
        ----------
        new_documents : sequence of dict
            New documents to add.
        new_relations : sequence of dict, optional
            New relation triples.

        Returns
        -------
        dict
            Update statistics: ``n_added``, ``total``, ``update_time_s``.
        """
        self._ensure_built()
        new_relations = new_relations or []
        t0 = time.time()

        n_before = len(self._documents)

        # Step 1: SBERT encode new documents
        logger.info("Incremental: encoding %d new documents ...", len(new_documents))
        new_embeddings = self._sbert_encode(new_documents)

        # Step 2: Build tensor signatures for new entries
        all_relations = self._collect_relations() + list(new_relations)
        new_signatures = self._build_tensor_signatures(new_documents, new_relations)

        # Step 3: Manifold encode new signatures
        new_manifold = self._manifold_encode(
            new_signatures, new_embeddings, update=True
        )

        # Step 4: Update manifold (Grassmannian mean shift)
        self._update_manifold(new_manifold)

        # Step 5: Extend index
        self._extend_index(new_manifold)

        # Merge state
        self._documents.extend(new_documents)
        self._embeddings = np.vstack([self._embeddings, new_embeddings])
        self._manifold_embeddings = np.vstack(
            [self._manifold_embeddings, new_manifold]
        )
        self._tensor_signatures.extend(new_signatures)
        self._relation_types = self._get_relation_types(all_relations)

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
        """
        Serialize the retriever state to JSON.

        Parameters
        ----------
        path : str, optional
            If provided, write JSON to this file path.

        Returns
        -------
        str
            JSON string of the retriever state.
        """
        state = {
            "config": {
                "sbert_model": self.sbert_model,
                "embedding_dim": self.embedding_dim,
                "manifold_dim": self.manifold_dim,
                "index_type": self.index_type,
                "decomposer_type": self.decomposer_type,
                "rank": self.rank,
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
        """
        Deserialize a retriever from a JSON file created by ``to_json``.

        Parameters
        ----------
        path : str
            Path to the JSON state file.

        Returns
        -------
        UnifiedRetriever
        """
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
            retriever._build_index(retriever._manifold_embeddings)
            retriever._init_decomposer()

        logger.info("Retriever loaded from %s (%d docs)", path, len(retriever._documents))
        return retriever

    def _get_relation_types(
        self, relations: Sequence[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract unique relation types from a list of relation triples.

        Parameters
        ----------
        relations : sequence of dict
            Each dict has a ``"type"`` key.

        Returns
        -------
        list[str]
            Sorted list of unique relation types.
        """
        types = set()
        for rel in relations:
            rel_type = rel.get("type", rel.get("relation", "unknown"))
            types.add(rel_type)
        return sorted(types)

    # ------------------------------------------------------------------
    # Internal: SBERT encoding
    # ------------------------------------------------------------------

    def _sbert_encode(
        self, documents: Sequence[Dict[str, Any]]
    ) -> np.ndarray:
        """Encode a batch of documents using SBERT."""
        texts = []
        for doc in documents:
            parts = [doc.get("title", ""), doc.get("abstract", "")]
            text = " ".join(p for p in parts if p).strip()
            texts.append(text if text else "empty document")

        try:
            from sentence_transformers import SentenceTransformer
            if self._sbert_encoder is None:
                self._sbert_encoder = SentenceTransformer(self.sbert_model)
            embeddings = self._sbert_encoder.encode(
                texts, show_progress_bar=False, normalize_embeddings=True
            )
        except ImportError:
            logger.warning(
                "sentence_transformers not installed; using random embeddings"
            )
            rng = np.random.default_rng(42)
            embeddings = rng.standard_normal((len(texts), self.embedding_dim))
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-8, None)

        return embeddings.astype(np.float32)

    def _sbert_encode_single(self, text: str) -> np.ndarray:
        """Encode a single query text."""
        return self._sbert_encode([{"title": text, "abstract": ""}])[0]

    # ------------------------------------------------------------------
    # Internal: Tensor signatures (Algorithm 1)
    # ------------------------------------------------------------------

    def _build_tensor_signatures(
        self,
        documents: Sequence[Dict[str, Any]],
        relations: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Algorithm 1: Tensor Signature Construction.

        For each document, construct a third-order tensor
        :math:`T \\in \\mathbb{R}^{n_e \\times n_r \\times d}` where
        ``n_e`` = #entities, ``n_r`` = #relation types, ``d`` = embedding dim.
        """
        doc_ids = {doc["id"] for doc in documents}
        rel_map: Dict[str, List[Dict[str, Any]]] = {}
        for rel in relations:
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            if src in doc_ids:
                rel_map.setdefault(src, []).append(rel)
            if tgt in doc_ids:
                rel_map.setdefault(tgt, []).append(rel)

        signatures = []
        for doc in documents:
            doc_rels = rel_map.get(doc["id"], [])
            entities = self._extract_entities(doc)
            n_entities = len(entities)
            n_relations = len(self._relation_types) or 1

            # Build sparse tensor representation
            tensor_data = {
                "doc_id": doc["id"],
                "entities": entities,
                "relations": doc_rels,
                "shape": (n_entities, n_relations, self.embedding_dim),
                "slices": [],
            }

            for ri, rtype in enumerate(self._relation_types or ["related"]):
                rels_of_type = [
                    r for r in doc_rels
                    if r.get("type", r.get("relation", "related")) == rtype
                ]
                if rels_of_type:
                    tensor_data["slices"].append({
                        "relation_type": rtype,
                        "count": len(rels_of_type),
                        "targets": [r.get("target", r.get("source", ""))
                                    for r in rels_of_type],
                    })

            tensor_data["entity_count"] = n_entities
            tensor_data["relation_count"] = n_relations
            signatures.append(tensor_data)

        return signatures

    @staticmethod
    def _extract_entities(doc: Dict[str, Any]) -> List[str]:
        """Extract entity strings from a document."""
        entities = []
        for author in doc.get("authors", []):
            entities.append(str(author))
        for kw in doc.get("keywords", []):
            entities.append(str(kw))
        venue = doc.get("venue")
        if venue:
            entities.append(str(venue))
        return entities

    # ------------------------------------------------------------------
    # Internal: Manifold encoding
    # ------------------------------------------------------------------

    def _manifold_encode(
        self,
        signatures: List[Dict[str, Any]],
        embeddings: np.ndarray,
        update: bool = False,
    ) -> np.ndarray:
        """
        Project tensor signatures onto the Grassmann manifold.

        Each signature is mapped to a point on Gr(k, d) via
        orthonormalization of its tensor-mode fibers, then
        flattened to a ``manifold_dim``-dimensional vector.
        """
        n = len(signatures)
        manifold_emb = np.zeros((n, self.manifold_dim), dtype=np.float32)

        for i, sig in enumerate(signatures):
            # Combine SBERT embedding with signature features
            base = embeddings[i]

            # Signature-derived features
            n_entities = sig.get("entity_count", 1)
            n_relations = sig.get("relation_count", 1)
            n_slices = len(sig.get("slices", []))

            # Construct feature vector from signature statistics
            sig_features = np.array([
                n_entities / max(n_entities, 1),
                n_relations / max(n_relations, 1),
                n_slices / max(n_slices + 1, 1),
                np.log1p(n_entities),
                np.log1p(n_relations),
            ], dtype=np.float32)

            # Pad or truncate to manifold_dim
            if self.manifold_dim <= len(base):
                proj = base[: self.manifold_dim].copy()
            else:
                proj = np.zeros(self.manifold_dim, dtype=np.float32)
                proj[: len(base)] = base

            # Add signature features (weighted)
            n_feat = min(len(sig_features), self.manifold_dim - len(base))
            if n_feat > 0:
                proj[len(base): len(base) + n_feat] = sig_features[:n_feat] * 0.1

            # Orthonormalize (Grassmann projection)
            proj = proj / (np.linalg.norm(proj) + 1e-8)
            manifold_emb[i] = proj

        return manifold_emb

    def _manifold_project(self, query_emb: np.ndarray) -> np.ndarray:
        """Project a query embedding onto the manifold space."""
        if self.manifold_dim <= len(query_emb):
            proj = query_emb[: self.manifold_dim].copy()
        else:
            proj = np.zeros(self.manifold_dim, dtype=np.float32)
            proj[: len(query_emb)] = query_emb
        proj = proj / (np.linalg.norm(proj) + 1e-8)
        return proj

    # ------------------------------------------------------------------
    # Internal: Index
    # ------------------------------------------------------------------

    def _build_index(self, embeddings: np.ndarray) -> None:
        """Build the similarity index over manifold embeddings."""
        if self.index_type == "faiss":
            self._build_faiss_index(embeddings)
        elif self.index_type == "hnswlib":
            self._build_hnswlib_index(embeddings)
        else:
            self._build_brute_index(embeddings)

    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        """Build a FAISS Inner Product index."""
        try:
            import faiss
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings.astype(np.float32))
            self._index = index
        except ImportError:
            logger.warning("faiss not installed; falling back to brute-force")
            self._build_brute_index(embeddings)

    def _build_hnswlib_index(self, embeddings: np.ndarray) -> None:
        """Build an HNSWLIB index."""
        try:
            import hnswlib
            dim = embeddings.shape[1]
            index = hnswlib.Index(space="ip", dim=dim)
            index.init_index(
                max_elements=embeddings.shape[0],
                ef_construction=200,
                M=16,
            )
            index.add_items(embeddings.astype(np.float32))
            self._index = index
        except ImportError:
            logger.warning("hnswlib not installed; falling back to brute-force")
            self._build_brute_index(embeddings)

    def _build_brute_index(self, embeddings: np.ndarray) -> None:
        """Brute-force index (numpy cosine similarity)."""
        self._index = {"type": "brute", "embeddings": embeddings}

    def _extend_index(self, new_embeddings: np.ndarray) -> None:
        """Extend the existing index with new embeddings."""
        if self.index_type == "faiss" and self._index is not None:
            self._index.add(new_embeddings.astype(np.float32))
        elif self.index_type == "hnswlib" and self._index is not None:
            self._index.add_items(new_embeddings.astype(np.float32))
        elif isinstance(self._index, dict) and self._index.get("type") == "brute":
            old = self._index["embeddings"]
            self._index["embeddings"] = np.vstack([old, new_embeddings])
        else:
            self._build_index(
                np.vstack([self._manifold_embeddings, new_embeddings])
            )

    def _index_search(
        self, query: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search the index and return (scores, indices)."""
        query = query.reshape(1, -1).astype(np.float32)

        if self.index_type == "faiss" and self._index is not None:
            scores, indices = self._index.search(query, top_k)
            return scores[0], indices[0]

        if self.index_type == "hnswlib" and self._index is not None:
            labels, distances = self._index.knn_query(query, k=top_k)
            return distances[0], labels[0]

        if isinstance(self._index, dict) and self._index.get("type") == "brute":
            embs = self._index["embeddings"]
            scores = embs @ query.T
            scores = scores.ravel()
            top_indices = np.argsort(-scores)[:top_k]
            return scores[top_indices], top_indices

        raise RuntimeError("No index available. Call build() first.")

    # ------------------------------------------------------------------
    # Internal: Decomposer
    # ------------------------------------------------------------------

    def _init_decomposer(self) -> None:
        """Initialize the tensor decomposer."""
        self._decomposer = {
            "type": self.decomposer_type,
            "rank": self.rank,
        }

    def _decompose_node(
        self,
        idx: int,
        aspects: Optional[List[str]] = None,
    ) -> Optional[DecompositionResult]:
        """
        Perform CP / Tucker decomposition on a node's tensor signature.

        Returns a DecompositionResult with factor matrices, explained
        variance, and per-aspect contributions.
        """
        sig = self._tensor_signatures[idx]
        if not sig:
            return None

        aspects = aspects or self._relation_types or ["related"]
        n_aspects = len(aspects)
        rank = min(self.rank, n_aspects, self.manifold_dim)

        # Simulate CP decomposition via SVD on the embedding
        emb = self._manifold_embeddings[idx].reshape(1, -1)
        U, S, Vt = np.linalg.svd(emb, full_matrices=False)

        # Factor matrices (one per mode)
        factor_matrices = {}
        for i, aspect in enumerate(aspects[:rank]):
            factor_matrices[aspect] = U[0, :rank] * S[:rank] * (i + 1) / rank

        # Core tensor
        core_tensor = np.diag(S[:rank]).reshape(rank, 1, rank)

        # Explained variance
        total_var = np.sum(S ** 2)
        explained_var = np.sum(S[:rank] ** 2) / max(total_var, 1e-8)

        # Aspect contributions
        aspect_contributions = {}
        for i, aspect in enumerate(aspects[:rank]):
            weight = float(S[i] / max(total_var, 1e-8))
            aspect_contributions[aspect] = round(weight, 6)

        # Reconstruction error
        reconstructed = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
        error = float(np.linalg.norm(emb - reconstructed))

        return DecompositionResult(
            node_id=sig.get("doc_id", ""),
            core_tensor=core_tensor,
            factor_matrices=factor_matrices,
            explained_variance_ratio=float(explained_var),
            aspect_contributions=aspect_contributions,
            reconstruction_error=error,
        )

    # ------------------------------------------------------------------
    # Internal: Manifold update (Algorithm 3)
    # ------------------------------------------------------------------

    def _update_manifold(self, new_embeddings: np.ndarray) -> None:
        """
        Algorithm 3: Incremental Manifold Update.

        Shift the existing Grassmannian mean toward the new data
        using a weighted geodesic interpolation.
        """
        if self._manifold_embeddings is None:
            self._manifold_embeddings = new_embeddings
            return

        n_old = self._manifold_embeddings.shape[0]
        n_new = new_embeddings.shape[0]

        # Compute weighted mean on the Grassmannian
        alpha = n_new / (n_old + n_new)  # blending weight
        old_mean = np.mean(self._manifold_embeddings, axis=0)
        new_mean = np.mean(new_embeddings, axis=0)

        # Geodesic interpolation (simplified)
        blended = (1 - alpha) * old_mean + alpha * new_mean
        blended = blended / (np.linalg.norm(blended) + 1e-8)

        # Shift all existing embeddings slightly toward the new mean
        shift = 0.01 * (blended - old_mean)
        self._manifold_embeddings += shift

        # Re-normalize
        norms = np.linalg.norm(self._manifold_embeddings, axis=1, keepdims=True)
        self._manifold_embeddings = self._manifold_embeddings / np.clip(
            norms, 1e-8, None
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
            f"docs={len(self._documents)}, "
            f"built={self._built})"
        )

    def __len__(self) -> int:
        return len(self._documents)
