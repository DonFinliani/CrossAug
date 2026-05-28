import argparse
import json
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"    
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENAI_API_KEY"] = "123"

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.misc_utils import string_to_bool




def configure_logging():
    logging.basicConfig(level=logging.INFO, force=True)
    # Keep tqdm progress bars stable by suppressing noisy HTTP client request logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def build_save_dir(dataset: str, save_dir: str) -> str:
    if save_dir == "outputs":
        return os.path.join(save_dir, dataset)
    return f"{save_dir}_{dataset}"


def validate_existing_outputs(save_dir: str, llm_name: str, embedding_name: str) -> str:
    llm_label = llm_name.replace("/", "_")
    embedding_label = embedding_name.replace("/", "_")
    working_dir = os.path.join(save_dir, f"{llm_label}_{embedding_label}")
    graph_path = os.path.join(working_dir, "graph.pickle")
    openie_path = os.path.join(save_dir, f"openie_results_ner_{llm_label}.json")

    missing_paths = [path for path in [working_dir, graph_path, openie_path] if not os.path.exists(path)]
    if missing_paths:
        missing_text = "\n".join(missing_paths)
        raise FileNotFoundError(
            "Hidden triplet mining needs an existing indexed HippoRAG workspace. "
            f"The following paths are missing:\n{missing_text}"
        )

    return working_dir


def save_hidden_triplet_summary(save_dir: str, llm_name: str, summary: dict):
    llm_label = llm_name.replace("/", "_")
    summary_path = os.path.join(save_dir, f"hidden_triplet_summary_{llm_label}.json")

    os.makedirs(save_dir, exist_ok=True)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("Hidden triplet summary saved to %s", summary_path)


def maybe_override(hidden_triplet_overrides: dict, key: str, value):
    if value is not None:
        hidden_triplet_overrides[key] = value


def parse_args():
    parser = argparse.ArgumentParser(description="Run hidden triplet mining on an existing HippoRAG index.")
    parser.add_argument("--dataset", type=str, default="musique", help="Dataset name used to build the existing index.")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Base save directory or the dataset output prefix.")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini", help="LLM name used by the existing index.")
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default="https://api.openai.com/v1",
        help="LLM base URL for relation mining.",
    )
    parser.add_argument(
        "--embedding_name",
        type=str,
        default="nvidia/NV-Embed-v2",
        help="Embedding model name used by the existing index.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for GNN training and hidden-triplet LLM completion.")
    parser.add_argument(
        "--embedding_base_url",
        type=str,
        default="http://0.0.0.0:8000/v1/embeddings",
        help="Embedding base URL used when new entities/facts are inserted.",
    )
    parser.add_argument(
        "--load_hidden_triplet_augmented_content",
        type=str,
        default=None,
        help="Whether to load previously saved hidden-triplet augmented graph/facts before running.",
    )
    parser.add_argument(
        "--hidden_triplet_gnn_hidden_dim",
        type=int,
        default=None,
        help="Hidden size of the PyG encoder.",
    )
    parser.add_argument(
        "--hidden_triplet_gnn_encoder_type",
        type=str,
        choices=["gcn", "graphsage", "gatv2", "rgcn"],
        default=None,
        help="GNN encoder architecture used for hidden-triplet link prediction.",
    )
    parser.add_argument(
        "--hidden_triplet_gnn_epochs",
        type=int,
        default=None,
        help="Training epochs for the PyG encoder.",
    )
    parser.add_argument(
        "--hidden_triplet_learning_rate",
        type=float,
        default=None,
        help="Learning rate for the PyG encoder.",
    )
    parser.add_argument(
        "--hidden_triplet_graph_refresh_interval",
        type=int,
        default=None,
        help="Refresh the dynamically remasked fact-edge split and support graph once every k epochs.",
    )
    parser.add_argument(
        "--hidden_triplet_early_stopping_patience",
        type=int,
        default=None,
        help="Stop GNN training after this many evaluation points without eval_loss improvement. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--hidden_triplet_early_stopping_min_delta",
        type=float,
        default=None,
        help="Minimum eval_loss decrease required to reset early stopping patience.",
    )
    parser.add_argument(
        "--hidden_triplet_max_chars_per_chunk",
        type=int,
        default=None,
        help="Maximum number of characters kept from each evidence chunk.",
    )
    parser.add_argument(
        "--hidden_triplet_llm_max_tokens",
        type=int,
        default=None,
        help="Maximum completion tokens for each hidden triplet mining LLM call.",
    )
    parser.add_argument("--hidden_triplet_subgraph_walk_length", type=int, default=None)
    parser.add_argument("--hidden_triplet_subgraph_walks_per_root", type=int, default=None)
    parser.add_argument("--hidden_triplet_subgraph_negative_walks_per_root", type=int, default=None)
    parser.add_argument("--hidden_triplet_subgraph_fact_walk_scale", type=float, default=None)
    parser.add_argument("--hidden_triplet_subgraph_synonym_walk_scale", type=float, default=None)
    parser.add_argument("--hidden_triplet_subgraph_walk_weight_clip", type=float, default=None)
    parser.add_argument(
        "--hidden_triplet_subgraph_train_roots_per_epoch",
        type=int,
        default=None,
        help="Fixed number of root entities to sample each training epoch.",
    )
    parser.add_argument(
        "--hidden_triplet_subgraph_eval_roots",
        type=int,
        default=None,
        help="Fixed number of root entities to sample for evaluation. Set 0 to disable.",
    )
    parser.add_argument("--hidden_triplet_subgraph_fact_mask_ratio", type=float, default=None)
    parser.add_argument("--hidden_triplet_subgraph_entity_delete_ratio", type=float, default=None)
    parser.add_argument("--hidden_triplet_subgraph_min_nodes", type=int, default=None)
    parser.add_argument("--hidden_triplet_subgraph_min_fact_edges", type=int, default=None)
    parser.add_argument("--hidden_triplet_subgraph_inference_root_budget", type=float, default=None)
    parser.add_argument("--hidden_triplet_subgraph_missing_score_threshold", type=float, default=None)
    parser.add_argument("--hidden_triplet_subgraph_llm_budget", type=float, default=None)
    parser.add_argument("--hidden_triplet_subgraph_overlap_threshold", type=float, default=None)
    parser.add_argument("--hidden_triplet_subgraph_max_chunks_per_prompt", type=int, default=None)
    parser.add_argument("--hidden_triplet_subgraph_max_known_triples_per_prompt", type=int, default=None)
    parser.add_argument("--hidden_triplet_subgraph_max_entities_per_prompt", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging()

    dataset_save_dir = build_save_dir(args.dataset, args.save_dir)
    working_dir = validate_existing_outputs(
        save_dir=dataset_save_dir,
        llm_name=args.llm_name,
        embedding_name=args.embedding_name,
    )

    hidden_triplet_overrides = {}
    if args.load_hidden_triplet_augmented_content is not None:
        hidden_triplet_overrides["load_hidden_triplet_augmented_content"] = string_to_bool(
            args.load_hidden_triplet_augmented_content
        )
    maybe_override(hidden_triplet_overrides, "hidden_triplet_gnn_hidden_dim", args.hidden_triplet_gnn_hidden_dim)
    maybe_override(hidden_triplet_overrides, "hidden_triplet_gnn_encoder_type", args.hidden_triplet_gnn_encoder_type)
    maybe_override(hidden_triplet_overrides, "hidden_triplet_gnn_epochs", args.hidden_triplet_gnn_epochs)
    maybe_override(hidden_triplet_overrides, "hidden_triplet_learning_rate", args.hidden_triplet_learning_rate)
    maybe_override(
        hidden_triplet_overrides,
        "hidden_triplet_graph_refresh_interval",
        args.hidden_triplet_graph_refresh_interval,
    )
    maybe_override(
        hidden_triplet_overrides,
        "hidden_triplet_early_stopping_patience",
        args.hidden_triplet_early_stopping_patience,
    )
    maybe_override(
        hidden_triplet_overrides,
        "hidden_triplet_early_stopping_min_delta",
        args.hidden_triplet_early_stopping_min_delta,
    )
    maybe_override(
        hidden_triplet_overrides,
        "hidden_triplet_max_chars_per_chunk",
        args.hidden_triplet_max_chars_per_chunk,
    )
    maybe_override(hidden_triplet_overrides, "hidden_triplet_llm_max_tokens", args.hidden_triplet_llm_max_tokens)
    for key in [
        "hidden_triplet_subgraph_walk_length",
        "hidden_triplet_subgraph_walks_per_root",
        "hidden_triplet_subgraph_negative_walks_per_root",
        "hidden_triplet_subgraph_fact_walk_scale",
        "hidden_triplet_subgraph_synonym_walk_scale",
        "hidden_triplet_subgraph_walk_weight_clip",
        "hidden_triplet_subgraph_train_roots_per_epoch",
        "hidden_triplet_subgraph_eval_roots",
        "hidden_triplet_subgraph_fact_mask_ratio",
        "hidden_triplet_subgraph_entity_delete_ratio",
        "hidden_triplet_subgraph_min_nodes",
        "hidden_triplet_subgraph_min_fact_edges",
        "hidden_triplet_subgraph_inference_root_budget",
        "hidden_triplet_subgraph_missing_score_threshold",
        "hidden_triplet_subgraph_llm_budget",
        "hidden_triplet_subgraph_overlap_threshold",
        "hidden_triplet_subgraph_max_chunks_per_prompt",
        "hidden_triplet_subgraph_max_known_triples_per_prompt",
        "hidden_triplet_subgraph_max_entities_per_prompt",
    ]:
        maybe_override(hidden_triplet_overrides, key, getattr(args, key))
    hidden_triplet_overrides.setdefault("load_hidden_triplet_augmented_content", False)

    # Reuse the same save_dir / model-name layout as the original index so HippoRAG
    # loads the stored graph, embeddings, and OpenIE cache instead of rebuilding them.
    config = BaseConfig(
        save_dir=dataset_save_dir,
        dataset=args.dataset,
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        seed=args.seed,
        embedding_model_name=args.embedding_name,
        enable_hidden_triplet_mining=False,
        **hidden_triplet_overrides,
    )

    logging.info("Loading HippoRAG workspace from %s", working_dir)
    hipporag = HippoRAG(
        global_config=config,
        embedding_base_url=args.embedding_base_url,
    )

    summary = hipporag.mine_hidden_triplets_with_gnn()
    save_hidden_triplet_summary(dataset_save_dir, args.llm_name, summary)
    logging.info("Hidden triplet mining finished: %s", summary)
    print(summary)


if __name__ == "__main__":
    main()
