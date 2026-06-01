# ADR-0003: Qdrant as Vector Store with Named Vectors

## Status

Proposed

## Context

v1 uses in-memory vector indices (FAISS/HNSWLIB/brute-force NumPy). These have no persistence, no distributed deployment, no metadata filtering, and no multi-vector support.

v2 requires:
1. **Persistence**: Data must survive process restarts
2. **Named vectors**: Four channels need independent vector storage (ADR-0001)
3. **Payload filtering**: Task-specific queries need metadata-based pre-filtering (e.g., filter by time range, venue, author)
4. **API-native**: The vector store must be accessible via API, not just in-process library calls
5. **Moderate deployment complexity**: Target users are AI companies, not HPC clusters

### Options Evaluated

| Criterion | Qdrant | Milvus | FAISS (v1) |
|-----------|--------|--------|-------------|
| Named vectors | ✅ Native | ⚠️ Multi-field (less elegant) | ❌ |
| Payload filtering | ✅ Rich JSON payloads | ✅ Scalar filtering | ❌ |
| Persistence | ✅ Built-in | ✅ Built-in | ❌ |
| Distributed | ✅ Distributed mode | ✅ Native distributed | ❌ |
| Deployment | Low (single binary / Docker) | High (etcd + MinIO + 3+ containers) | Lowest (pip install) |
| Python client | ✅ Official async/sync | ✅ Official | ✅ faiss-cpu |
| Community | Active, growing | Very active | Mature, stable |

## Decision

**Adopt Qdrant as the primary vector store.**

Key reasons:
1. **Named vectors** map directly to the four-channel architecture — each channel is a named vector within the same collection
2. **Low deployment complexity** — single Docker container or single binary, suitable for AI companies' dev environments
3. **Rich payload filtering** — supports complex boolean filters on metadata, essential for task-specific queries
4. **Official async Python client** — aligns with API service architecture

FAISS remains available as a **fallback for local development/testing** (no Docker required).

## Consequences

### Positive
- Four channels stored as named vectors in a single Qdrant collection
- Metadata filtering enables task-specific pre-filtering without re-encoding
- Persistent storage eliminates need for index rebuild on restart
- Distributed mode available for production scaling

### Negative
- Introduces a runtime dependency (Qdrant server process)
- Slight latency increase vs. in-process FAISS (~2-5ms network overhead)
- Migration path from v1's in-memory indices requires re-indexing all documents

### Mitigations
- Provide a Docker Compose config for one-command local deployment
- Keep FAISS as an in-memory backend option for testing
- Write a migration script to re-index v1 data into Qdrant
