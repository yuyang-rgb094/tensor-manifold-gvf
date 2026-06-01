# ADR-0006: Channel-Level Weight Adjustability via Task-Specific Attention Heads

## Status

Proposed

## Context

v1 uses a single vector space for all retrieval tasks. The user identified that different tasks (author disambiguation, citation intent recognition, school clustering, keyword PCA) require different emphasis on different information modalities. The current system cannot adjust the relative importance of semantic content vs. topology vs. metadata vs. temporal signals.

With the four-channel architecture (ADR-0001) and cross-modal attention fusion (ADR-0002), we need a mechanism for task-specific channel weighting.

### Options Evaluated

1. **Channel-level weight vectors**: Simple learned weight vector `[w_sem, w_meta, w_topo, w_time]` per task. Lightweight but too coarse — can't capture channel interactions.

2. **Task-specific attention heads**: Each task has its own set of cross-attention parameters. The attention weights naturally reflect per-task channel importance. More expressive, captures channel interactions.

3. **Independent subspaces per task**: Each task has a completely separate vector space. Most flexible but doubles/quadruples storage and compute.

4. **Hybrid (core tasks independent, secondary shared)**: Core tasks (semantic retrieval, citation analysis) get independent subspaces; secondary tasks share a common space. Balanced but architecturally complex.

## Decision

**Adopt task-specific attention heads (Option 2).**

All tasks share a single unified vector space. Each task has its own cross-attention head parameters that learn to weight the four channels differently. The unified space avoids storage/compute explosion while still providing task-adaptive channel weighting.

Formally:
- Shared: channel embeddings $h_1, h_2, h_3, h_4$
- Per-task: attention head parameters $\theta_t$ for task $t$
- Task output: $o_t = \text{CrossAttn}(h_1, h_2, h_3, h_4; \theta_t)$
- Storage: only $\theta_t$ per task (small), not entire vector space

## Consequences

### Positive
- Single unified vector space — storage-efficient
- Task-specific channel weighting — addresses the core problem
- Attention weights are interpretable (can inspect which channels each task attends to)
- Adding new tasks only requires training new attention head parameters

### Negative
- Shared space means tasks compete — a representation good for retrieval may be suboptimal for disambiguation
- Attention head training requires task-specific labeled data or at least task-specific feedback signals
- The "single unified space" assumption may break down for sufficiently different tasks

### Mitigations
- Start with the four priority tasks (semantic retrieval, citation analysis, author disambiguation, trend analysis) and validate unified-space quality
- If unified space proves insufficient for a specific task, that task can be upgraded to an independent subspace (migration path to Option 3)
- Use multi-task learning to train attention heads jointly, sharing channel encoder parameters
