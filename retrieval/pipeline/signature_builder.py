"""
Tensor Signature Construction (Algorithm 1).

Builds third-order tensor signatures from entity-relation triples.
Each signature captures the multi-modal structure of a document
as a tensor :math:`T \\in \\mathbb{R}^{n_e \\times n_r \\times d}`.

Extracted from ``retriever.py`` for modular pipeline usage.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SignatureBuilder:
    """Construct tensor signatures for a collection of documents.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the embedding vectors used in the tensor
        third mode (default ``384`` for ``all-MiniLM-L6-v2``).
    """

    def __init__(self, embedding_dim: int = 384) -> None:
        self.embedding_dim = embedding_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_relation_types(self, relations: List[Dict[str, Any]]) -> List[str]:
        """Extract unique sorted relation types from a list of relation triples.

        Parameters
        ----------
        relations : list[dict]
            Each dict must contain a ``"type"`` key (falls back to
            ``"relation"`` and then ``"unknown"``).

        Returns
        -------
        list[str]
            Alphabetically sorted list of unique relation types.
        """
        types: set[str] = set()
        for rel in relations:
            rel_type = rel.get("type", rel.get("relation", "unknown"))
            types.add(rel_type)
        return sorted(types)

    def build(
        self,
        documents: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build tensor signatures for all documents.

        Algorithm 1: Tensor Signature Construction.

        For each document a third-order tensor
        :math:`T \\in \\mathbb{R}^{n_e \\times n_r \\times d}` is
        constructed where ``n_e`` = number of entities, ``n_r`` = number
        of relation types, and ``d`` = embedding dimensionality.

        Parameters
        ----------
        documents : list[dict]
            Each dict must have at least ``id``.  May also include
            ``authors``, ``keywords``, ``venue`` (used for entity
            extraction).
        relations : list[dict]
            Each dict: ``{"source": str, "target": str, "type": str}``.

        Returns
        -------
        list[dict]
            One signature dict per document with keys:

            - ``doc_id``          -- document identifier
            - ``entities``        -- list of extracted entity strings
            - ``relations``       -- list of relation dicts for this doc
            - ``shape``           -- ``(n_entities, n_relations, embedding_dim)``
            - ``slices``          -- per-relation-type slice descriptors
            - ``entity_count``    -- number of entities
            - ``relation_count``  -- number of relation types
        """
        relation_types = self.get_relation_types(relations)

        # Build a doc-id -> relations lookup
        doc_ids = {doc["id"] for doc in documents}
        rel_map: Dict[str, List[Dict[str, Any]]] = {}
        for rel in relations:
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            if src in doc_ids:
                rel_map.setdefault(src, []).append(rel)
            if tgt in doc_ids:
                rel_map.setdefault(tgt, []).append(rel)

        signatures: List[Dict[str, Any]] = []
        for doc in documents:
            doc_rels = rel_map.get(doc["id"], [])
            entities = self.extract_entities(doc)
            n_entities = len(entities)
            n_relations = len(relation_types) or 1

            # Build sparse tensor representation
            tensor_data: Dict[str, Any] = {
                "doc_id": doc["id"],
                "entities": entities,
                "relations": doc_rels,
                "shape": (n_entities, n_relations, self.embedding_dim),
                "slices": [],
            }

            for ri, rtype in enumerate(relation_types or ["related"]):
                rels_of_type = [
                    r for r in doc_rels
                    if r.get("type", r.get("relation", "related")) == rtype
                ]
                if rels_of_type:
                    tensor_data["slices"].append({
                        "relation_type": rtype,
                        "count": len(rels_of_type),
                        "targets": [
                            r.get("target", r.get("source", ""))
                            for r in rels_of_type
                        ],
                    })

            tensor_data["entity_count"] = n_entities
            tensor_data["relation_count"] = n_relations
            signatures.append(tensor_data)

        logger.info(
            "Built %d tensor signatures (relation_types=%s, dim=%d)",
            len(signatures),
            relation_types,
            self.embedding_dim,
        )
        return signatures

    @staticmethod
    def extract_entities(doc: Dict[str, Any]) -> List[str]:
        """Extract entity strings from a document.

        Collects ``authors``, ``keywords``, and ``venue`` as entity
        strings.

        Parameters
        ----------
        doc : dict
            Document dict with optional ``authors``, ``keywords``, and
            ``venue`` keys.

        Returns
        -------
        list[str]
            Concatenated list of entity strings.
        """
        entities: List[str] = []
        for author in doc.get("authors", []):
            entities.append(str(author))
        for kw in doc.get("keywords", []):
            entities.append(str(kw))
        venue = doc.get("venue")
        if venue:
            entities.append(str(venue))
        return entities
