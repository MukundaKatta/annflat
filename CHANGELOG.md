# Changelog

## [0.1.0] - 2026-05-09

### Added
- `Index::new(metric)` for cosine, L2, or inner-product similarity.
- `add(id, vec)` and `add_batch(ids, matrix)`.
- `search(query, k)` and parallel `search_batch(queries, k, parallel)`.
- abi3-py310 wheel: one wheel for CPython 3.10 through 3.13.
