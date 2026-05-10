# Changelog

## [0.1.1] - 2026-05-10

### Added
- `Index::remove(id)` — O(n) swap-remove of the first matching entry.
  Returns `true` if found, `false` otherwise. `dim` is preserved.
- `Index::save(path)` / `Index::load(path)` — JSON persistence of the
  full index (metric, dim, ids, vectors). Round-trip-safe; binary
  serialization deferred to v0.2.

### Errors
- New `AnnFlatError::Io` and `AnnFlatError::Serde` variants surfaced by
  `save`/`load`.

## [0.1.0] - 2026-05-09

### Added
- `Index::new(metric)` for cosine, L2, or inner-product similarity.
- `add(id, vec)` and `add_batch(ids, matrix)`.
- `search(query, k)` and parallel `search_batch(queries, k, parallel)`.
- abi3-py310 wheel: one wheel for CPython 3.10 through 3.13.
