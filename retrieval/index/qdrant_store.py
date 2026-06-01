"""Qdrant Vector Store adapter for four-channel named vectors.

Stores each channel (semantic, metadata, topology, temporal) as a named
vector in a single Qdrant collection, with rich payload for metadata filtering.

See ADR-0003 for architectural rationale.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QdrantStore:
    """Qdrant adapter for four-channel named vector storage.

    Each document is stored as a single Qdrant point with:
    - 4 named vectors (semantic, metadata, topology, temporal)
    - Rich payload (id, title, authors, year, venue, keywords)

    Qdrant requires point IDs to be UUIDs or integers. This adapter
    maintains an internal doc_id -> UUID mapping and stores the
    original doc_id in the payload for retrieval.

    Parameters
    ----------
    collection_name : str
        Qdrant collection name.
    host : str
        Qdrant server host. Use ``":memory:"`` for in-memory testing.
    port : int
        Qdrant server port.
    api_key : str, optional
        API key for Qdrant Cloud.
    distance : str
        Distance metric: ``"cosine"``, ``"dot"``, or ``"euclid"``.
    """

    def __init__(
        self,
        collection_name: str = "tensor_manifold_gvf",
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        distance: str = "cosine",
    ):
        from qdrant_client import QdrantClient

        self.collection_name = collection_name
        self._host = host
        self._port = port
        self._distance = distance

        # Connect to Qdrant
        if host == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(
                host=host,
                port=port,
                api_key=api_key,
            )

        self._channel_dims: Dict[str, int] = {}
        # doc_id -> UUID mapping (Qdrant requires UUID or int point IDs)
        self._id_to_uuid: Dict[str, str] = {}
        # UUID -> doc_id reverse mapping
        self._uuid_to_id: Dict[str, str] = {}

    def create_collection(
        self,
        channel_dims: Dict[str, int],
        recreate: bool = True,
    ) -> None:
        """Create the Qdrant collection with named vectors for each channel.

        Parameters
        ----------
        channel_dims : dict
            Mapping of channel name to vector dimension.
            e.g. ``{"semantic": 1024, "metadata": 256, ...}``
        recreate : bool
            If True, delete and recreate the collection.
        """
        from qdrant_client import models

        self._channel_dims = channel_dims

        distance_map = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclid": models.Distance.EUCLID,
        }
        distance = distance_map.get(self._distance, models.Distance.COSINE)

        vectors_config = {
            name: models.VectorParams(size=dim, distance=distance)
            for name, dim in channel_dims.items()
        }

        if recreate:
            # Suppress deprecation warning for recreate_collection
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                )
        else:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
            )

        # Reset ID mappings on collection recreation
        self._id_to_uuid = {}
        self._uuid_to_id = {}

        logger.info(
            "Created Qdrant collection '%s' with %d named vectors: %s",
            self.collection_name,
            len(channel_dims),
            list(channel_dims.keys()),
        )

    def _get_or_create_uuid(self, doc_id: str) -> str:
        """Get existing UUID for doc_id or create a new one."""
        if doc_id not in self._id_to_uuid:
            new_uuid = str(uuid.uuid4())
            self._id_to_uuid[doc_id] = new_uuid
            self._uuid_to_id[new_uuid] = doc_id
        return self._id_to_uuid[doc_id]

    def upsert(
        self,
        documents: List[Dict[str, Any]],
        channel_embeddings: Dict[str, np.ndarray],
    ) -> None:
        """Upsert documents with four-channel named vectors.

        Parameters
        ----------
        documents : list of dict
            Each dict must have at least ``id``.
        channel_embeddings : dict
            Mapping of channel name to embedding array of shape (N, D).
        """
        from qdrant_client import models

        n_docs = len(documents)
        points = []

        for i, doc in enumerate(documents):
            doc_id = doc.get("id", f"doc_{i}")
            point_uuid = self._get_or_create_uuid(doc_id)

            # Build named vectors dict
            vectors = {}
            for channel_name, emb_array in channel_embeddings.items():
                if i < len(emb_array):
                    vectors[channel_name] = emb_array[i].tolist()

            # Build payload (original doc_id stored here)
            payload = {
                "id": doc_id,
                "title": doc.get("title", ""),
                "abstract": doc.get("abstract", ""),
                "authors": doc.get("authors", []),
                "year": doc.get("year"),
                "venue": doc.get("venue", ""),
                "keywords": doc.get("keywords", []),
            }

            points.append(
                models.PointStruct(
                    id=point_uuid,
                    vector=vectors,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        logger.info(
            "Upserted %d documents into '%s'",
            n_docs,
            self.collection_name,
        )

    def search(
        self,
        query_vector: np.ndarray,
        channel: str = "semantic",
        top_k: int = 10,
        filter_year: Optional[Tuple[int, int]] = None,
        filter_venue: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search by a specific named vector channel.

        Parameters
        ----------
        query_vector : np.ndarray
            Query vector of shape (1, D) or (D,).
        channel : str
            Named vector channel to search (semantic/metadata/topology/temporal).
        top_k : int
            Number of results to return.
        filter_year : (int, int), optional
            Year range filter [min_year, max_year].
        filter_venue : str, optional
            Venue name filter (case-insensitive substring match).

        Returns
        -------
        list of dict
            Each dict has ``id``, ``score``, and ``payload``.
            ``id`` is the original document ID (not UUID).
        """
        from qdrant_client import models

        # Ensure query is 1-D
        query = query_vector.ravel().tolist()

        # Build filter
        qdrant_filter = None
        conditions = []

        if filter_year is not None:
            conditions.append(
                models.FieldCondition(
                    key="year",
                    range=models.Range(
                        gte=filter_year[0],
                        lte=filter_year[1],
                    ),
                )
            )

        if filter_venue is not None:
            conditions.append(
                models.FieldCondition(
                    key="venue",
                    match=models.MatchValue(value=filter_venue),
                )
            )

        if conditions:
            qdrant_filter = models.Filter(must=conditions)

        # Search
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query,
            using=channel,
            limit=top_k,
            query_filter=qdrant_filter,
        ).points

        # Map back to original doc IDs from payload
        return [
            {
                "id": (r.payload or {}).get("id", str(r.id)),
                "score": r.score,
                "payload": r.payload or {},
            }
            for r in results
        ]

    def delete(self, ids: List[str]) -> None:
        """Delete documents by original document ID.

        Parameters
        ----------
        ids : list of str
            Document IDs to delete.
        """
        from qdrant_client import models

        # Map doc IDs to UUIDs
        uuids = []
        for doc_id in ids:
            if doc_id in self._id_to_uuid:
                point_uuid = self._id_to_uuid.pop(doc_id)
                self._uuid_to_id.pop(point_uuid, None)
                uuids.append(point_uuid)

        if not uuids:
            logger.warning("No matching UUIDs found for delete: %s", ids)
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=uuids
            ),
        )

        logger.info("Deleted %d documents from '%s'", len(uuids), self.collection_name)

    def count(self) -> int:
        """Return the number of documents in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0

    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        collections = self.client.get_collections().collections
        return any(c.name == self.collection_name for c in collections)
