# CrossAug

CrossAug contains the main experiment code for three RAG frameworks:

- `HippoRAG2/`: HippoRAG2 with CrossAug-based hidden-triplet subgraph completion.
- `LightRAG/`: LightRAG with CrossAug-based hidden-triplet subgraph completion.
- `gfm-rag/`: GFM-RAG QA over graphs converted from HippoRAG2 base/CrossAug graphs.

The three frameworks should be reproduced in three separate virtual environments. This avoids dependency conflicts between HippoRAG2, LightRAG, and GFM-RAG. Each framework directory includes a `requirements.txt` exported from the corresponding local conda environment that successfully ran the main experiments; use that file as the primary Python dependency lock for reproduction.

## Repository Layout

```text
CrossAug/
  artifacts/                 # compressed cache artifacts with LFS
  HippoRAG2/                  # HippoRAG2 main experiment code
  LightRAG/                  # LightRAG main experiment code
  gfm-rag/                   # GFM-RAG main experiment code
```


## Common Runtime Settings

Use an OpenAI-compatible LLM endpoint and an OpenAI-compatible embedding endpoint.

Example values used in our runs:

```bash
export LLM_BASE_URL=http://127.0.0.1:9500/v1
export LLM_NAME=Models/Qwen3-32B
export EMBEDDING_NAME=BAAI-bge-m3
export EMBEDDING_BASE_URL=http://127.0.0.1:8000/v1/embeddings
export OPENAI_API_KEY=not_needed
```

## Cache Restore

The repository ships the HippoRAG2 cache with LFS as:

```text
artifacts/hipporag_outputs_cache.tar.zst
```

Restore it from the repository root:

```bash
tar --zstd -xf artifacts/hipporag_outputs_cache.tar.zst -C HippoRAG2
```

This recreates:

```text
HippoRAG2/outputs/
```

GFM-RAG is experimented on base/CrossAug graphs produced by HippoRAG2+CrossAug. Rebuild it from the restored HippoRAG2 outputs using the conversion command in the GFM-RAG section.



## 1. HippoRAG2 Environment

Create a dedicated environment and install the frozen dependencies exported from the verified HippoRAG2 conda environment:

```bash
conda create -n crossaug-hipporag2 python=3.10 -y
conda activate crossaug-hipporag2
cd /path/to/CrossAug/HippoRAG2
pip install -r requirements.txt
pip install -e . --no-deps
```

Run LiteraryQA base and CrossAug flows over the first 50 books:

```bash
python main.py \
  --dataset literaryqa \
  --llm_base_url "$LLM_BASE_URL" \
  --llm_name "$LLM_NAME" \
  --embedding_name "$EMBEDDING_NAME" \
  --embedding_base_url "$EMBEDDING_BASE_URL" \
  --enable_hidden_triplet_mining false \
  --load_hidden_triplet_augmented_content false \
  --literaryqa_book_limit 50

python main.py \
  --dataset literaryqa \
  --llm_base_url "$LLM_BASE_URL" \
  --llm_name "$LLM_NAME" \
  --embedding_name "$EMBEDDING_NAME" \
  --embedding_base_url "$EMBEDDING_BASE_URL" \
  --enable_hidden_triplet_mining true \
  --load_hidden_triplet_augmented_content true \
  --literaryqa_book_limit 50
```

Run the multi-hop datasets:

```bash
for dataset in musique hotpotqa 2wikimultihopqa; do
  python main.py \
    --dataset "$dataset" \
    --llm_base_url "$LLM_BASE_URL" \
    --llm_name "$LLM_NAME" \
    --embedding_name "$EMBEDDING_NAME" \
    --embedding_base_url "$EMBEDDING_BASE_URL" \
    --enable_hidden_triplet_mining false \
    --load_hidden_triplet_augmented_content false

  python main.py \
    --dataset "$dataset" \
    --llm_base_url "$LLM_BASE_URL" \
    --llm_name "$LLM_NAME" \
    --embedding_name "$EMBEDDING_NAME" \
    --embedding_base_url "$EMBEDDING_BASE_URL" \
    --enable_hidden_triplet_mining true \
    --load_hidden_triplet_augmented_content true
done
```

HippoRAG2 outputs are written under:

```text
HippoRAG2/outputs/<dataset>/
```

The CrossAug reproduction profile is recorded in result metadata as `hidden_triplet_reproduction_profile`.

## 2. LightRAG Environment

Create a dedicated environment and install the frozen dependencies exported from the verified LightRAG conda environment:

```bash
conda create -n crossaug-lightrag python=3.10 -y
conda activate crossaug-lightrag
cd /path/to/CrossAug/LightRAG
pip install -r requirements.txt
pip install -e ".[gnn]" --no-deps
```

Configure the LLM and embedding endpoints:

```bash
export LLM_BINDING_HOST="$LLM_BASE_URL"
export LLM_MODEL="$LLM_NAME"
export LLM_BINDING_API_KEY=not_needed

export EMBEDDING_BINDING_HOST=http://127.0.0.1:8000/v1
export EMBEDDING_MODEL=BAAI-bge-m3
export EMBEDDING_BINDING_API_KEY=not_needed
```

Run LiteraryQA:

```bash
python examples/lightrag_literaryqa_gnn_eval.py \
  --literaryqa_manifest ../HippoRAG2/reproduce/dataset/literaryqa/manifest.json \
  --output_dir outputs/lightrag_literaryqa_gnn \
  --book_limit 50 \
  --run_base \
  --run_gnn
```

Run the multi-hop datasets:

```bash
python examples/lightrag_multihopqa_gnn_eval.py \
  --dataset_root ../HippoRAG2/reproduce/dataset \
  --datasets musique hotpotqa 2wikimultihopqa \
  --output_dir outputs/lightrag_multihopqa_gnn \
  --run_base \
  --run_gnn
```

LightRAG outputs are written under:

```text
LightRAG/outputs/lightrag_literaryqa_gnn/
LightRAG/outputs/lightrag_multihopqa_gnn/
```

The CrossAug reproduction profile is recorded in the CrossAug artifact metadata as `reproduction_profile`.

## 3. GFM-RAG Environment

Create a dedicated environment and install the frozen dependencies exported from the verified GFM-RAG conda environment. GFM-RAG requires Python 3.12.

```bash
conda create -n crossaug-gfmrag python=3.12 -y
conda activate crossaug-gfmrag
cd /path/to/CrossAug/gfm-rag
pip install -r requirements.txt
pip install -e . --no-deps
```

Restore the HippoRAG2 cache first, then convert HippoRAG2 outputs into GFM-RAG stage1 data:

```bash
python scripts/convert_hipporag_outputs_to_stage1.py \
  --hipporag-outputs ../HippoRAG2/outputs \
  --qa-source-dir ../HippoRAG2/reproduce/dataset \
  --literaryqa-source-dir ../HippoRAG2/reproduce/dataset/literaryqa/books \
  --output-root data/hipporag_stage1_exports \
  --literaryqa-mode per-book
```

Run GFM-RAG QA:

```bash
export OPENAI_BASE_URL="$LLM_BASE_URL"
export LLM="$LLM_NAME"
export N_GPU=2
export DATA_ROOT=data/hipporag_stage1_exports

bash scripts/base_gnn.bash
```

GFM-RAG reads converted HippoRAG2 base/CrossAug stage1 data and writes its own stage2/stage3 outputs under the GFM-RAG working directories.

## Reproduction Notes

- The `requirements.txt` files are generated from working local conda environments, so they should reproduce the Python package side of the setup more reliably than reinstalling from broad package specs. They still cannot fully lock external conditions such as CUDA driver compatibility, system libraries, GPU availability, or whether the LLM/embedding endpoints are reachable.
- If a frozen requirement line uses a machine-local `file:///...` build path and fails on another machine, replace that line with the same package/version from PyPI or conda before rerunning `pip install -r requirements.txt`.
- For cached HippoRAG2 reproduction, keep `force_index_from_scratch=false` and `force_openie_from_scratch=false`, which are the defaults.
- Changing any of the four exposed CrossAug parameters intentionally changes the CrossAug run configuration and may trigger new hidden-triplet mining outputs.
- LiteraryQA main results in the paper use the first 50 books; use `--literaryqa_book_limit 50` or `--book_limit 50` accordingly.
- If your endpoint enforces lower concurrency, reduce HippoRAG2 `--llm_concurrency`, LightRAG `--query_concurrency`, or GFM-RAG `N_THREAD`.
