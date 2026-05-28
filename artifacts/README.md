# Cache Artifacts

This directory contains compressed cache artifacts that are small enough to ship with the repository.

## HippoRAG

- `hipporag_outputs_cache.tar.zst`: compressed `HippoRAG/outputs` cache for the main experiments.
- The archive intentionally excludes answer-judge caches because the paper experiments do not require LLM-as-Judge metrics.

Restore it from the repository root:

```bash
tar --zstd -xf artifacts/hipporag_outputs_cache.tar.zst -C HippoRAG
```

After extraction, `HippoRAG/outputs` will be available for cache reuse.

## LightRAG

LightRAG outputs are not provided in the repository because the compressed cache is still large. A local non-released backup may be kept outside the repository as:

```text
/home/CrossAug_external_artifacts/lightrag_outputs_cache.not_provided.tar.zst
```
