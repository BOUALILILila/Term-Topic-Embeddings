# coding=utf-8
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments as BaseTrainingArgs

@dataclass
class Base:
    def __post_init__(self):
        pass

@dataclass
class ModelArguments(Base):
    """
    Arguments for model/config/tokenizer.
    """
    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models, checkpoint of a SenseEmbed model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    token_dim: int = field(default=None)
    num_senses: int = field(default=3, metadata={"help": "Number of senses per token"})
    num_sense_attn_heads: int = field(default=None, metadata={"help": "Number of attention heads used for attention over senses"})
    init_embed: str = field(default='uniform', metadata={"help": "Initialization function for the sense embedding layer"})
    positionless_core: bool = field(default=False, metadata={"help": "Discard position embeddings for the contextualized transformer model core"})
    similarity_fct: str = field(
        default='dot', metadata={"help": "The similarity function to use among ('dot','cos')"}
    )
    alpha_mmse: float = field(
        default=1., metadata={"help": "The contribution of the margin_mse loss to the final loss."}
    )
    alpha_emb_loss: float = field(
        default=1., metadata={"help": "The contribution of the embedding loss to the final loss."}
    )
    emb_loss: str = field(
        default=None, metadata={"help": "The embedding loss to use among ('cos','mse')"}
    )

    pool_raw_scores: bool = field(default=False, metadata={"help": "True: pool raw attention scores=> one softmax"})

    sense_temp: float = field(default=1., metadata={"help": "Softmax temperature for SSCA layer"})

    sense_embed_init: str = field(default=None, metadata={"help": "Path to sense embeddings initialization if any."})
    sense_embed_init_norm: bool = field(default=False, metadata={"help": "apply L2 norm on kmeans centroid vectors when init"})

    def __post_init__(self):
        super().__post_init__()
        self.reduce_dim = False if self.token_dim is None else True

@dataclass
class PosModelArguments(ModelArguments):
    num_position_attn_heads: int = field(default=None, metadata={"help": "Number of attention heads used for attention over positions"})
    sliding_context_window: int = field(default=3, metadata={"help": "The context window of positions to consider around the token (odd)"})
    use_value: bool = field(default=False, metadata={"help": "Use the value layer for position selfattention AttnSGU"})
    sgu_circulant_matrix: bool = field(default=False, metadata={"help": "Construxt a Toeplitz-like matrix as the W weight in SGU"}) # compat
    is_cross: bool = field(default=True, metadata={"help": "AttnSGUFromTransformer, True: K comes from the SE reps else from the transformer reps"})
    add_positional_embeddings: bool = field(default=False, metadata={"help": "Add positional embeddings to SRM output"})
    
    def __post_init__(self):
        super().__post_init__()
        if self.sliding_context_window > 0:
            assert self.sliding_context_window % 2 == 1, " The sliding_context_window must be odd since it includes the central token"
            self.sliding_context_window = int((self.sliding_context_window-1) /2)

@dataclass
class DataArguments:
    train_path: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    max_seq_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    min_seq_len: int = field(
        default=0,
        metadata={
            "help": "The minimum total input sequence length after tokenization for passage. Sequences shorter "
                    "than this will be discarded."
        },
    )
    min_count: int = field(
        default=2, metadata={"help": "Min token occurrence"}
    )
    sample: float = field(
        default=1e-3, metadata={"help": "Frequent tokens subsampling"}
    )
    rng_seed: int = field(
        default=1234, metadata={"help": "Random generator seed for choosing central tokens"}
    )
    N: int = field(
        default=50, metadata={"help": "Number of central tokens to sample from each passage"}
    )
    context_window: int = field(
        default=5, metadata={"help": "Context window size, in terms of tokens"}
    )
    trim_rule: str = field(
        default=None, metadata={"help": "Custom function to decide whether to keep or discard this word."
                                        "If a custom `trim_rule` is not specified, the default behaviour is simply `count >= min_count`."}
    )
    sampling_strategy: str = field(
        default='uni', metadata={"help": "For EmbKD loss either use frequency subsampling or a uniform distribution ('frq','uni','none'), 'none' uses all tokens in a sequence."}
    )

    def __post_init__(self):
        if self.context_window > 0:
            assert self.context_window % 2 == 1, " The sliding_context_window must be odd since it includes the central token"
            self.context_window = int((self.context_window-1) /2)

@dataclass
class TrainingArguments(BaseTrainingArgs):
    freeze_transformer_weights: bool = field(
        default=False, metadata={"help": "Freeze the transformer model weights during training."}
    )
    log_gradients: bool = field(default=False)


@dataclass
class TripletsDataArguments:

    data_folder: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
   
    max_seq_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_query_len: int = field(
        default=50,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    rng_seed: int = field(
        default=1234, metadata={"help": "Random generator seed for choosing central tokens"}
    )
    max_passages: int = field(
        default=0, metadata={"help": "Maximum number of passages to use for training"}
    )
    negs_to_use: str = field(
        default=None, metadata={"help": "From which systems should negatives be used? Multiple systems seperated by comma. None = all"}
    )
    num_negs_per_system: int = field(
        default=5, metadata={"help": "Number of negatives from each system"}
    )
    use_all_queries: bool = field(
        default=False, metadata={"help": "Use all queries even if the query has no positive and no negative passages"}
    )
