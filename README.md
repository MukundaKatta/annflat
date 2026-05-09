# annflat

Small in-memory flat-file ANN over f32 vectors. Cosine, L2, or inner product.
Rust core, Python frontend.

## When to use it

For vector counts up to ~100k where flat brute-force is fast enough and the
operational overhead of Pinecone, Qdrant, etc. is not worth it. Re-rankers,
caches, prototypes, in-process retrieval over a fixed corpus.

For larger corpora or write-heavy workloads, use a real ANN index (HNSW
via `usearch`, IVF via `faiss`, or a hosted service).

## Install

```bash
pip install annflat
```

## Quickstart

```python
import numpy as np
from annflat import Index, Metric

idx = Index(metric=Metric.COSINE)

# Bulk add 10k 384-d embeddings.
ids = [f"doc-{i}" for i in range(10_000)]
vectors = np.random.randn(10_000, 384).astype(np.float32)
idx.add_batch(ids, vectors)

# Single query.
q = np.random.randn(384).astype(np.float32)
hits = idx.search(q, k=10)
for h in hits:
    print(h.id, h.score)

# Batch query, parallelized.
queries = np.random.randn(100, 384).astype(np.float32)
results = idx.search_batch(queries, k=10, parallel=True)
```

## License

Dual-licensed under MIT or Apache-2.0.
