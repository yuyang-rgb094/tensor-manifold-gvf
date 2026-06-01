# ADR-0005: Knowledge API Service Architecture

## Status

Proposed

## Context

v1 is a Python library with CLI scripts. Users run `build_index.py` to build indices and `query_demo.py` for interactive queries. This is suitable for researchers running local experiments but not for AI agents or production services.

v2 targets AI Researchers, Builders, and Agent technology companies. They need:
1. **Programmatic access**: REST/GraphQL API, not CLI
2. **Structured output**: JSON responses suitable for agent consumption
3. **Task-specific endpoints**: Different retrieval strategies for different tasks
4. **Composable**: The API should be composable into larger agent workflows (e.g., as an MCP server or LangChain tool)

## Decision

**Design v2 as a Knowledge API service with the following layers:**

### API Layer (FastAPI)
- REST endpoints for each task type:
  - `POST /api/v1/search` — Semantic retrieval (paper/section/knowledge-unit granularity)
  - `POST /api/v1/citation/network` — Citation network analysis
  - `POST /api/v1/authors/disambiguate` — Author entity disambiguation
  - `POST /api/v1/trends/evolution` — Research trend & concept evolution
  - `POST /api/v1/knowledge-graph` — Knowledge graph construction
  - `POST /api/v1/insights` — Insight extraction & inspiration discovery
- All endpoints accept JSON, return JSON
- OpenAPI schema auto-generated for agent integration

### Task Router
- Parses incoming request to determine task type and required channels
- Selects appropriate task-specific attention head
- Routes to cross-modal fusion with task-specific weights

### Fusion Engine
- Cross-modal attention fusion (ADR-0002)
- Four-channel encoding (ADR-0001)
- Outputs unified task-specific embeddings

### Storage Layer
- Qdrant for vector storage (ADR-0003)
- Neo4j or NebulaGraph for graph storage (citation edges, author-paper relations)
- Redis for caching frequently accessed results

### Data Ingestion Pipeline
- OAG format input (backward compatible with v1)
- Incremental update support
- Four-channel parallel encoding during ingestion

## Consequences

### Positive
- AI agents can directly consume the API
- Task-specific endpoints provide clear, predictable interfaces
- OpenAPI schema enables auto-generated client SDKs
- Can be deployed as a standalone service or embedded in larger systems

### Negative
- Significant engineering effort to build and maintain the API layer
- Introduces operational complexity (service deployment, monitoring, scaling)
- Requires authentication, rate limiting, and other production concerns

### Mitigations
- Start with FastAPI (minimal boilerplate, auto-docs)
- Use Docker Compose for local development
- Design the API layer as a thin wrapper over the existing core library
- Provide an MCP server adapter for direct Agent integration
