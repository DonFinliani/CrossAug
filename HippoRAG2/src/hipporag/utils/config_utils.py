import os
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Literal,
    Union,
    Optional
)

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class HippoRAGHiddenTripletProfileRegistry:
    """Dataset-specific hidden-triplet profiles used by reproduction runs."""

    common_profile: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_triplet_gnn_hidden_dim": 256,
        "hidden_triplet_gnn_encoder_type": "rgcn",
        "hidden_triplet_gnn_epochs": 200,
        "hidden_triplet_learning_rate": 2e-3,
        "hidden_triplet_graph_refresh_interval": 10,
        "hidden_triplet_early_stopping_patience": 50,
        "hidden_triplet_early_stopping_min_delta": 1e-4,
        "hidden_triplet_max_chars_per_chunk": 4500,
        "hidden_triplet_llm_max_tokens": 12000,
        "hidden_triplet_subgraph_walk_length": 3,
        "hidden_triplet_subgraph_fact_walk_scale": 1.0,
        "hidden_triplet_subgraph_synonym_walk_scale": 0.5,
        "hidden_triplet_subgraph_walk_weight_clip": 5.0,
        "hidden_triplet_subgraph_train_roots_per_epoch": 128,
        "hidden_triplet_subgraph_eval_roots": 64,
        "hidden_triplet_subgraph_fact_mask_ratio": 0.2,
        "hidden_triplet_subgraph_entity_delete_ratio": 0.08,
        "hidden_triplet_subgraph_min_nodes": 4,
        "hidden_triplet_subgraph_min_fact_edges": 1,
        "hidden_triplet_subgraph_overlap_threshold": 0.5,
        "hidden_triplet_subgraph_max_chunks_per_prompt": 15,
        "hidden_triplet_subgraph_max_known_triples_per_prompt": 120,
        "hidden_triplet_subgraph_max_entities_per_prompt": 120,
    })
    dataset_profiles: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "literaryqa": {
            "hidden_triplet_subgraph_walks_per_root": 50,
            "hidden_triplet_subgraph_negative_walks_per_root": 60,
            "hidden_triplet_subgraph_inference_root_budget": 512,
            "hidden_triplet_subgraph_missing_score_threshold": 0.5,
            "hidden_triplet_subgraph_llm_budget": 100,
        },
        "musique": {
            "hidden_triplet_subgraph_walks_per_root": 20,
            "hidden_triplet_subgraph_negative_walks_per_root": 30,
            "hidden_triplet_subgraph_inference_root_budget": 0.1,
            "hidden_triplet_subgraph_missing_score_threshold": 0.5,
            "hidden_triplet_subgraph_llm_budget": 0.1,
        },
        "hotpotqa": {
            "hidden_triplet_subgraph_walks_per_root": 20,
            "hidden_triplet_subgraph_negative_walks_per_root": 30,
            "hidden_triplet_subgraph_inference_root_budget": 0.1,
            "hidden_triplet_subgraph_missing_score_threshold": 0.5,
            "hidden_triplet_subgraph_llm_budget": 0.1,
        },
        "2wikimultihopqa": {
            "hidden_triplet_subgraph_walks_per_root": 20,
            "hidden_triplet_subgraph_negative_walks_per_root": 30,
            "hidden_triplet_subgraph_inference_root_budget": 0.1,
            "hidden_triplet_subgraph_missing_score_threshold": 0.4,
            "hidden_triplet_subgraph_llm_budget": 0.1,
        },
    })
    default_dataset: str = "literaryqa"

    @staticmethod
    def normalize_dataset_name(dataset_name: str) -> str:
        normalized = str(dataset_name or "").lower()
        if normalized in {"2wiki", "2wikimultihopqa"}:
            return "2wikimultihopqa"
        return normalized

    def get_profile(self, dataset_name: str) -> Dict[str, Any]:
        profile_dataset = self.normalize_dataset_name(dataset_name)
        profile = dict(self.common_profile)
        profile.update(
            self.dataset_profiles.get(
                profile_dataset,
                self.dataset_profiles[self.default_dataset],
            )
        )
        return profile


HIPPORAG_HIDDEN_TRIPLET_PROFILE_REGISTRY = HippoRAGHiddenTripletProfileRegistry()


def normalize_hipporag_hidden_triplet_profile_dataset_name(dataset_name: str) -> str:
    return HIPPORAG_HIDDEN_TRIPLET_PROFILE_REGISTRY.normalize_dataset_name(dataset_name)


def get_hipporag_hidden_triplet_profile(dataset_name: str) -> Dict[str, Any]:
    return HIPPORAG_HIDDEN_TRIPLET_PROFILE_REGISTRY.get_profile(dataset_name)


@dataclass
class BaseConfig:
    """One and only configuration."""
    # LLM specific attributes 
    llm_name: str = field(
        default="gpt-4o-mini",
        metadata={"help": "Class name indicating which LLM model to use."}
    )
    llm_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for the LLM model, if none, means using OPENAI service."}
    )
    embedding_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for an OpenAI compatible embedding model, if none, means using OPENAI service."}
    )
    azure_endpoint: str = field(
        default=None,
        metadata={"help": "Azure Endpoint URI for the LLM model, if none, uses OPENAI service directly."}
    )
    azure_embedding_endpoint: str = field(
        default=None,
        metadata={"help": "Azure Endpoint URI for the OpenAI embedding model, if none, uses OPENAI service directly."}
    )
    max_new_tokens: Union[None, int] = field(
        default=2048,
        metadata={"help": "Max new tokens to generate in each inference."}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={"help": "How many chat completion choices to generate for each input message."}
    )
    seed: Union[None, int] = field(
        default=42,
        metadata={"help": "Random seed. Set to None only if you intentionally want backend-side nondeterministic generation."}
    )
    temperature: float = field(
        default=0,
        metadata={"help": "Temperature for sampling in each inference."}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: { "type": "json_object" },
        metadata={"help": "Specifying the format that the model must output."}
    )
    
    ## LLM specific attributes -> Async hyperparameters
    max_retry_attempts: int = field(
        default=5,
        metadata={"help": "Max number of retry attempts for an asynchronous API calling."}
    )
    llm_concurrency: int = field(
        default=32,
        metadata={"help": "Unified concurrency limit for all non-answer-judge LLM calls in the main HippoRAG flow, including OpenIE, QA generation, and hidden-triplet completion."}
    )
    llm_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional shared directory for non-answer-judge LLM cache. Defaults to <save_dir>/llm_cache."}
    )
    # Storage specific attributes
    force_openie_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing openie files and rebuild them from scratch."}
    )

    # Storage specific attributes 
    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing storage files and graph data and will rebuild from scratch."}
    )
    rerank_dspy_file_path: str = field(
        default=None,
        metadata={"help": "Path to the rerank dspy file."}
    )
    passage_node_weight: float = field(
        default=0.05,
        metadata={"help": "Multiplicative factor that modified the passage node weights in PPR."}
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "If set to True, will save the OpenIE model to disk."}
    )
    
    # Preprocessing specific attributes
    text_preprocessor_class_name: str = field(
        default="TextPreprocessor",
        metadata={"help": "Name of the text-based preprocessor to use in preprocessing."}
    )
    preprocess_encoder_name: str = field(
        default="gpt-4o",
        metadata={"help": "Name of the encoder to use in preprocessing (currently implemented specifically for doc chunking)."}
    )
    preprocess_chunk_overlap_token_size: int = field(
        default=128,
        metadata={"help": "Number of overlap tokens between neighbouring chunks."}
    )
    preprocess_chunk_max_token_size: int = field(
        default=None,
        metadata={"help": "Max number of tokens each chunk can contain. If set to None, the whole doc will treated as a single chunk."}
    )
    preprocess_chunk_func: Literal["by_token", "by_word"] = field(default='by_token')
    
    
    # Information extraction specific attributes
    information_extraction_model_name: Literal["openie_openai_gpt", ] = field(
        default="openie_openai_gpt",
        metadata={"help": "Class name indicating which information extraction model to use."}
    )
    openie_mode: Literal["offline", "online"] = field(
        default="online",
        metadata={"help": "Mode of the OpenIE model to use."}
    )
    skip_graph: bool = field(
        default=False,
        metadata={"help": "Whether to skip graph construction or not. Set it to be true when running vllm offline indexing for the first time."}
    )
    
    
    # Embedding specific attributes
    embedding_model_name: str = field(
        default="nvidia/NV-Embed-v2",
        metadata={"help": "Class name indicating which embedding model to use."}
    )
    embedding_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size of calling embedding model."}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "Max sequence length for the embedding model."}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto",
        metadata={"help": "Data type for local embedding model."}
    )
    
    
    
    # Graph construction specific attributes
    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "k for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={"help": "Batch size for query embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={"help": "Batch size for key embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "Similarity threshold to include candidate synonymy nodes."}
    )
    is_directed_graph: bool = field(
        default=False,
        metadata={"help": "Whether the graph is directed or not."}
    )
    
    
    
    # Retrieval specific attributes
    linking_top_k: int = field(
        default=5,
        metadata={"help": "The number of linked nodes at each retrieval step"}
    )
    retrieval_top_k: int = field(
        default=200,
        metadata={"help": "Retrieving k documents at each step"}
    )
    damping: float = field(
        default=0.5,
        metadata={"help": "Damping factor for ppr algorithm."}
    )
    
    
    # QA specific attributes
    max_qa_steps: int = field(
        default=1,
        metadata={"help": "For answering a single question, the max steps that we use to interleave retrieval and reasoning."}
    )
    qa_top_k: int = field(
        default=5,
        metadata={"help": "Feeding top k documents to the QA model for reading."}
    )
    enable_answer_judge: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate QA answers with an LLM-based semantic consistency judge."}
    )
    answer_judge_model: str = field(
        default="deepseek-v4-pro",
        metadata={"help": "OpenAI-compatible model name used by the LLM answer judge."}
    )
    answer_judge_base_url: str = field(
        default="https://api.deepseek.com",
        metadata={"help": "OpenAI-compatible base URL used by the LLM answer judge."}
    )
    answer_judge_api_key_env: str = field(
        default="DEEPSEEK_API_KEY",
        metadata={"help": "Environment variable name containing the LLM answer judge API key."}
    )
    answer_judge_max_workers: int = field(
        default=32,
        metadata={"help": "Maximum number of concurrent requests for the LLM answer judge."}
    )
    answer_judge_max_tokens: int = field(
        default=5120,
        metadata={"help": "Maximum output tokens for each LLM answer judge request."}
    )
    answer_judge_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional shared directory for answer-judge LLM cache. Defaults to <save_dir>/llm_cache/answer_judge."}
    )
    
    # Save dir (highest level directory)
    save_dir: str = field(
        default=None,
        metadata={"help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."}
    )
    
    
    
    # Dataset running specific attributes
    ## Dataset running specific attributes -> General
    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa', 'literaryqa']] = field(
        default=None,
        metadata={"help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."}
    )
    ## Dataset running specific attributes -> Graph
    graph_type: Literal[
        'dpr_only', 
        'entity', 
        'passage_entity', 'relation_aware_passage_entity',
        'passage_entity_relation', 
        'facts_and_sim_passage_node_unidirectional',
    ] = field(
        default="facts_and_sim_passage_node_unidirectional",
        metadata={"help": "Type of graph to use in the experiment."}
    )
    corpus_len: Optional[int] = field(
        default=None,
        metadata={"help": "Length of the corpus to use."}
    )

    # Hidden triplet mining specific attributes
    enable_hidden_triplet_mining: bool = field(
        default=False,
        metadata={"help": "Whether to run GNN-based hidden triplet mining after indexing."}
    )
    force_hidden_triplet_mining_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, ignore existing hidden-triplet mining artifacts and rebuild them from the base graph."}
    )
    load_hidden_triplet_augmented_content: bool = field(
        default=True,
        metadata={"help": "Whether to load hidden-triplet augmented graph and fact/entity stores during retrieval."}
    )
    hidden_triplet_reproduction_profile_name: Optional[str] = field(
        default=None,
        metadata={"help": "Named hidden-triplet reproduction profile applied by the main experiment runner."}
    )
    hidden_triplet_reproduction_profile: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Resolved hidden-triplet reproduction parameters for audit only; not used in cache matching."}
    )
    hidden_triplet_gnn_hidden_dim: int = field(
        default=256,
        metadata={"help": "Hidden dimension for the lightweight GNN encoder used in hidden triplet mining."}
    )
    hidden_triplet_gnn_encoder_type: str = field(
        default="rgcn",
        metadata={"help": "GNN encoder architecture for hidden triplet mining. Options: gcn, graphsage, gatv2, rgcn."}
    )
    hidden_triplet_gnn_epochs: int = field(
        default=200,
        metadata={"help": "Training epochs for the hidden triplet mining GNN."}
    )
    hidden_triplet_learning_rate: float = field(
        default=2e-3,
        metadata={"help": "Learning rate for hidden triplet mining GNN training."}
    )
    hidden_triplet_graph_refresh_interval: int = field(
        default=10,
        metadata={"help": "Refresh the dynamically remasked fact-edge split and support graph once every k epochs during hidden triplet GNN training."}
    )
    hidden_triplet_early_stopping_patience: int = field(
        default=50,
        metadata={"help": "Stop hidden triplet GNN training after this many evaluation points without eval_loss improvement. Set <= 0 to disable."}
    )
    hidden_triplet_early_stopping_min_delta: float = field(
        default=1e-4,
        metadata={"help": "Minimum eval_loss decrease required to reset hidden triplet GNN early stopping patience."}
    )
    hidden_triplet_max_chars_per_chunk: int = field(
        default=4500,
        metadata={"help": "Maximum number of characters to keep from each evidence chunk before sending it to the LLM."}
    )
    hidden_triplet_llm_max_tokens: int = field(
        default=12000,
        metadata={"help": "Maximum number of completion tokens for each hidden triplet mining LLM call."}
    )
    hidden_triplet_subgraph_walk_length: int = field(
        default=3,
        metadata={"help": "Number of entity-node hops used to sample hidden-triplet subgraphs. Traversing chunk nodes does not consume this budget."}
    )
    hidden_triplet_subgraph_walks_per_root: int = field(
        default=50,
        metadata={"help": "Number of weighted random walks launched from each sampled root entity."}
    )
    hidden_triplet_subgraph_negative_walks_per_root: int = field(
        default=60,
        metadata={"help": "Number of weighted random walks used to build clean negative subgraphs during subgraph missingness training."}
    )
    hidden_triplet_subgraph_fact_walk_scale: float = field(
        default=1.0,
        metadata={"help": "Sampling weight multiplier for fact edges during hidden-triplet weighted random walks."}
    )
    hidden_triplet_subgraph_synonym_walk_scale: float = field(
        default=0.5,
        metadata={"help": "Sampling weight multiplier for synonym edges during hidden-triplet weighted random walks."}
    )
    hidden_triplet_subgraph_walk_weight_clip: float = field(
        default=5.0,
        metadata={"help": "Maximum raw edge weight used when building weighted-random-walk transition weights."}
    )
    hidden_triplet_subgraph_train_roots_per_epoch: int = field(
        default=128,
        metadata={"help": "Fixed number of root entities sampled per epoch for subgraph missingness GNN training."}
    )
    hidden_triplet_subgraph_eval_roots: int = field(
        default=64,
        metadata={"help": "Fixed number of root entities sampled for subgraph missingness GNN evaluation. Set to 0 to disable eval roots."}
    )
    hidden_triplet_subgraph_fact_mask_ratio: float = field(
        default=0.2,
        metadata={"help": "Ratio of fact edges removed when constructing relation-missing subgraph views."}
    )
    hidden_triplet_subgraph_entity_delete_ratio: float = field(
        default=0.08,
        metadata={"help": "Ratio of eligible entity nodes removed when constructing entity-missing subgraph views."}
    )
    hidden_triplet_subgraph_min_nodes: int = field(
        default=4,
        metadata={"help": "Minimum sampled subgraph node count retained for subgraph missingness training/inference."}
    )
    hidden_triplet_subgraph_min_fact_edges: int = field(
        default=1,
        metadata={"help": "Minimum fact-edge count retained for sampled subgraph training/inference."}
    )
    hidden_triplet_subgraph_inference_root_budget: float = field(
        default=512,
        metadata={"help": "If this value is an integer >= 1, use it directly as the maximum sampled root entities scored during subgraph missingness inference. If it is a fractional value in (0, 1), resolve the budget as int(value * num_entity_nodes)."}
    )
    hidden_triplet_subgraph_missing_score_threshold: float = field(
        default=0.5,
        metadata={"help": "Minimum GNN missingness score required before a sampled subgraph can be sent to the LLM."}
    )
    hidden_triplet_subgraph_llm_budget: float = field(
        default=100,
        metadata={"help": "If this value is an integer >= 1, use it directly as the maximum number of high-missingness subgraphs sent to the LLM. If it is a fractional value in (0, 1), resolve the budget as int(value * num_chunks * 2)."}
    )
    hidden_triplet_subgraph_overlap_threshold: float = field(
        default=0.5,
        metadata={"help": "Maximum node-Jaccard overlap allowed between selected LLM subgraphs."}
    )
    hidden_triplet_subgraph_max_chunks_per_prompt: int = field(
        default=15,
        metadata={"help": "Maximum evidence chunks included in one subgraph-completion LLM prompt."}
    )
    hidden_triplet_subgraph_max_known_triples_per_prompt: int = field(
        default=120,
        metadata={"help": "Maximum known triples included in one subgraph-completion LLM prompt."}
    )
    hidden_triplet_subgraph_max_entities_per_prompt: int = field(
        default=120,
        metadata={"help": "Maximum existing entity strings included in one subgraph-completion LLM prompt."}
    )
    def __post_init__(self):
        if self.save_dir is None: # If save_dir not given
            if self.dataset is None: self.save_dir = 'outputs' # running freely
            else: self.save_dir = os.path.join('outputs', self.dataset) # customize your dataset's output dir here
        logger.debug(f"Initializing the highest level of save_dir to be {self.save_dir}")
