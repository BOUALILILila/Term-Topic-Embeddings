# coding=utf-8
import contextlib
from typing import Tuple, Any, Optional

import torch
from torch import nn
from transformers import (
    PreTrainedModel, 
    BatchEncoding,
)

from transformers.utils import logging


from arguments import ModelArguments
from model_utils import *

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

# ECIR 2023 Paper naming
# The Term Topic Module (TTM): represent token occurences as a weighted average of their sub-embeddings
# TTM = SenseEmbed class 
# sub-embeddings = sense_embed layer
# Sub-embeddings weighted average attention scoring in SenseDisambiguationUnit Module
class SenseEmbed(BaseModel):
    def __init__(
        self, 
        context_rep_model: PreTrainedModel, 
        model_args: ModelArguments,
        **kwargs,
        ):

        super().__init__()
        self.model_args = model_args
        self.freeze_transformer_weights = kwargs.pop('freeze_transformer_weights', False)

        self.transformer_model: PreTrainedModel = context_rep_model
        if self.freeze_transformer_weights:
            self.transformer_context = torch.no_grad
        else:
            self.transformer_context = contextlib.nullcontext
        
        if self.model_args.reduce_dim:
            self.tok_proj = nn.Linear(self.transformer_model.config.hidden_size, self.model_args.token_dim)
            self.init_weights(self.tok_proj)
        else:
            self.model_args.token_dim = self.transformer_model.config.hidden_size

        sense_embed_init = model_args.sense_embed_init
        if self.model_args.init_embed == 'uniform':
            sense_embeds = torch.zeros((self.transformer_model.config.vocab_size,
                                        self.model_args.num_senses,
                                        self.model_args.token_dim)).float()
            nn.init.uniform_(sense_embeds, -1.0, 1.0)
            if sense_embed_init is not None:
                logger.info("using kmeans init")
                if os.path.exists(sense_embed_init):
                    file_names = os.listdir(sense_embed_init)
                    if model_args.sense_embed_init_norm:
                        for tok_id in range(sense_embeds.shape[0]):
                            tok_init_embeddings = f"tok_{tok_id}_centroids.pt"
                            if tok_init_embeddings in file_names:
                                sense_embeds[tok_id] = nn.functional.normalize(
                                    torch.tensor(
                                        torch.load(os.path.join(sense_embed_init, tok_init_embeddings))
                                    ), p=2, dim=-1
                                )
                    else:
                        for tok_id in range(sense_embeds.shape[0]):
                            tok_init_embeddings = f"tok_{tok_id}_centroids.pt"
                            if tok_init_embeddings in file_names:
                                sense_embeds[tok_id] = torch.tensor(
                                    torch.load(os.path.join(sense_embed_init, tok_init_embeddings))
                                )
        elif self.model_args.init_embed == 'xavier':
            sense_embeds = self.init_embedding(self.transformer_model.config.vocab_size,
                                                self.model_args.num_senses,
                                                self.model_args.token_dim)
        elif self.model_args.init_embed == 'xavierk':
            sense_embeds = torch.zeros((self.transformer_model.config.vocab_size,
                                        self.model_args.num_senses,
                                        self.model_args.token_dim)).float()
            nn.init.xavier_uniform_(sense_embeds)
        else:
            raise ValueError(f"Unknown init {self.model_args.init_embed}")

        self.sense_embeds = nn.Parameter(sense_embeds, requires_grad=True)

        # Bertconfig for cross attention
        sdu_config = None
        self.sdu = SenseDisambiguationUnit(
            self.model_args.pool_raw_scores,
            self.model_args.num_sense_attn_heads, 
            dropout_prob=0., hidden_size=self.model_args.token_dim,
            config=sdu_config,
            temperature=self.model_args.sense_temp)

        logger.info(f" >> Attention over senses uses {self.sdu.attn.num_attention_heads} attention heads.")
        logger.info(f" >> Sense Embedding Layer initialized following {self.model_args.init_embed}")

    def init_embedding(self, vocab_size, num_senses, token_dim):
        layers = [torch.zeros((vocab_size, token_dim)).float() for _ in range(num_senses)]
        for layer in layers:
            torch.nn.init.xavier_uniform_(layer)

        stacked = torch.stack(layers, dim=1)
        assert stacked.shape == (vocab_size, num_senses, token_dim)
        return stacked
        
    def embed(self, output_transformer_embs=False, **features):
        # Contextual embeddings using a pre-trained Transformer model
        with self.transformer_context():
            if isinstance(self.transformer_model, DistilBertModel):
                output_dict = self.transformer_model(
                    input_ids=features['input_ids'], attention_mask=features['attention_mask'], 
                    return_dict=True
                )
            else:
                # assert all([x in features for x in ['input_ids', 'attention_mask', 'token_type_ids']])
                output_dict = self.transformer_model(**features, return_dict=True) 
        # Dimension reduction
        if self.model_args.reduce_dim:
            tok_reps = self.tok_proj(output_dict.last_hidden_state) # bs, max_seq_len, token_dim
        else:
            tok_reps = output_dict.last_hidden_state
        
         # Sense Disambiguation
        # # Sense Embeddings from static embedding layer
        tok_sense_embeds = self.sense_embeds[features['input_ids']] # bs, seq_len, num_senses, tok_dim
        # # Sense disambiguation using cross attention
        sdu_outputs = self.sdu(tok_reps, tok_sense_embeds) # sense disambiguated representation| bs, seq_len, tok_dim
        
        return SeqOutput(
            token_emb = sdu_outputs[0],
            transformer_token_emb = tok_reps if output_transformer_embs else None,
        )

    def embed_avg(self, output_transformer_embs=False, **features):
        # Contextual embeddings using a pre-trained Transformer model
        with self.transformer_context():
            if isinstance(self.transformer_model, DistilBertModel):
                output_dict = self.transformer_model(
                    input_ids=features['input_ids'], attention_mask=features['attention_mask'], 
                    return_dict=True
                )
            else:
                # assert all([x in features for x in ['input_ids', 'attention_mask', 'token_type_ids']])
                output_dict = self.transformer_model(**features, return_dict=True) 
        # Dimension reduction
        if self.model_args.reduce_dim:
            tok_reps = self.tok_proj(output_dict.last_hidden_state) # bs, max_seq_len, token_dim
        else:
            tok_reps = output_dict.last_hidden_state
        
         # Sense Disambiguation
        # # Sense Embeddings from static embedding layer
        tok_sense_embeds = self.sense_embeds[features['input_ids']] # bs, seq_len, num_senses, tok_dim
        # # Sense disambiguation using cross attention
        # sdu_outputs = self.sdu(tok_reps, tok_sense_embeds) # sense disambiguated representation| bs, seq_len, tok_dim
        
        return SeqOutput(
            token_emb = tok_sense_embeds.mean(-2), # mean avg of senses 
            transformer_token_emb = tok_reps if output_transformer_embs else None,
        )
        
    def forward_representation(self, 
            transformer_inputs: BatchEncoding, 
            sample_token_ind: torch.Tensor,
            output_transformer_embs: bool = False,
            output_sense_attention_weights: bool = False,
            **kwargs: Any
        ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:  
        """
        Parameters
        ----------
        transformer_inputs : BatchEncoding
            Dict with input_ids and attention_mask.
        sample_token_ind : torch.Tensor of shape (bs,N)
            Indices of the sampled random tokens.
        
        Returns
        -------
        sample_token_emb: torch.tensor of shape (bs*N, token_dim)
            The embedding of the random tokens.
        """
        # Contextual embeddings using a pre-trained Transformer model
        with self.transformer_context():
            if isinstance(self.transformer_model, DistilBertModel):
                output_dict = self.transformer_model(
                    input_ids=transformer_inputs['input_ids'], attention_mask=transformer_inputs['attention_mask'], 
                    return_dict=True
                )
            else:
                # assert all([x in transformer_inputs for x in ['input_ids', 'attention_mask', 'token_type_ids']])
                output_dict = self.transformer_model(**transformer_inputs, return_dict=True) 
        # Dimension reduction
        if self.model_args.reduce_dim:
            tok_reps = self.tok_proj(output_dict.last_hidden_state) # bs*max_seq_len*token_dim
        else:
            tok_reps = output_dict.last_hidden_state

        sample_word_index = sample_token_ind.unsqueeze(2).expand(sample_token_ind.shape[0], sample_token_ind.shape[1], tok_reps.shape[2])
        sample_word_hs = torch.gather(tok_reps, 1, sample_word_index) # bs, N, tok_dim

        sample_word_ids = torch.gather(transformer_inputs['input_ids'], 1, sample_token_ind)  # bs, N
        
        sample_emb_senses = self.sense_embeds[sample_word_ids] # bs, N, num_senses, tok_dim
        
        # Handle empty batches -> no sampled pairs | cannot ahppen but just in case
        if len(sample_word_hs) == 0 :
            raise ValueError('No central token samples in the current batch')

        # Attention over senses
        out = self.sdu(
            sample_word_hs, sample_emb_senses, 
            output_attentions=output_sense_attention_weights)# bs, N, tok_dim
        tok_dim = tok_reps.shape[-1]
        return RandomSampleOutput(
            sample_token_emb = out[0].view(-1, tok_dim),
            transformer_sample_token_emb = sample_word_hs.view(-1, tok_dim) if output_transformer_embs else None,
            sense_attention_weights = out[1] if output_sense_attention_weights else None,
        )

# ECIR 2023 Paper naming
# The Local Context Module (LCM): Refines TTM produced embeddings with local contextulization (sliding window attention layer)
# TTM-LCM = SenseEmbedwithAttnSGUFromTransformer class 
# sub-embeddings = sense_embed layer
# Sub-embeddings weighted average attention scoring in SenseDisambiguationUnit Module
# AttnSpatialGatingUnit variants are used for local contextualization
# In the paper we use CrossAttnSpatialGatingUnit
class SenseEmbedwithAttnSGUFromTransformer(BaseModel):
    def __init__(
        self, 
        context_rep_model: PreTrainedModel, 
        model_args: ModelArguments,
        **kwargs,
    ) -> None:

        super().__init__()
        self.model_args = model_args
        self.freeze_transformer_weights = kwargs.pop('freeze_transformer_weights', False)

        self.transformer_model: PreTrainedModel = context_rep_model
        if self.freeze_transformer_weights:
            self.transformer_context = torch.no_grad
        else:
            self.transformer_context = contextlib.nullcontext
        
        if self.model_args.reduce_dim:
            self.tok_proj = nn.Linear(self.transformer_model.config.hidden_size, self.model_args.token_dim)
            self.init_weights(self.tok_proj)
        else:
            self.model_args.token_dim = self.transformer_model.config.hidden_size

        if self.model_args.init_embed == 'uniform':
            sense_embeds = torch.zeros((self.transformer_model.config.vocab_size,
                                        self.model_args.num_senses,
                                        self.model_args.token_dim)).float()
            nn.init.uniform_(sense_embeds, -1.0, 1.0)
        elif self.model_args.init_embed == 'xavier':
            sense_embeds = self.init_embedding(self.transformer_model.config.vocab_size,
                                                self.model_args.num_senses,
                                                self.model_args.token_dim)
        else:
            raise ValueError(f"Unknown init {self.model_args.init_embed}")

        self.sense_embeds = nn.Parameter(sense_embeds, requires_grad=True)

        # Bertconfig for cross attention
        sdu_config = None
        self.sdu = SenseDisambiguationUnit(
            self.model_args.pool_raw_scores,
            self.model_args.num_sense_attn_heads, 
            dropout_prob=0., hidden_size=self.model_args.token_dim,
            config=sdu_config,
            temperature=self.model_args.sense_temp)

        if self.model_args.add_positional_embeddings:
            logger.info(" >> Using positional embeddings with layernorm and dropout")
            self.add_pos_embeddings = EmbeddingLayerWithPoisition(self.transformer_model.config, init=self.transformer_model.embeddings.position_embeddings)

        # Bertconfig for position attention        
        sgu_config = None
        if self.model_args.is_cross:
            self.sgu = CrossAttnSpatialGatingUnit(
                self.model_args.sliding_context_window,
                self.model_args.num_position_attn_heads,
                dropout_prob=0., hidden_size=self.model_args.token_dim,
                config=sgu_config,
                use_value=self.model_args.use_value,
                query_input_dim=self.model_args.token_dim, # query is derived from the transformer representations 
                )
        else:
            self.sgu = AttnSpatialGatingUnitFromTransformer(
                self.model_args.sliding_context_window,
                self.model_args.num_position_attn_heads,
                dropout_prob=0., hidden_size=self.model_args.token_dim,
                config=sgu_config,
                use_value=self.model_args.use_value,
                )

        # Projection layer
        self.channel_proj = nn.Linear(self.model_args.token_dim, self.model_args.token_dim)
        self.init_weights(self.channel_proj)
        
        
        logger.info(f" >> Attention over senses uses {self.sdu.attn.num_attention_heads} attention heads.")
        logger.info(f" >> Attention over positions uses {self.sgu.attn.num_attention_heads} attention heads.")
        logger.info(f" >> Attention over positions uses Value layer : {self.model_args.use_value}.")
        logger.info(f" >> Query for position signal is derived from the transformer representations.")
    
    def forward_representation(self, 
            transformer_inputs: BatchEncoding, 
            sample_token_ind: Optional[torch.Tensor] = None,
            output_transformer_embs: bool = False,
            output_sense_attention_weights: bool = False,
            **kwargs: Any
        ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:  
        """
        Parameters
        ----------
        transformer_inputs : BatchEncoding
            Dict with input_ids and attention_mask.
        sample_token_ind : torch.Tensor of shape (bs,N)
            Indices of the sampled random tokens.
        
        Returns
        -------
        sample_token_emb: torch.tensor of shape (bs*N, token_dim)
            The embedding of the random tokens.
        """
        # Contextual embeddings using a pre-trained Transformer model
        with self.transformer_context():
            if isinstance(self.transformer_model, DistilBertModel):
                output_dict = self.transformer_model(
                    input_ids=transformer_inputs['input_ids'], attention_mask=transformer_inputs['attention_mask'], 
                    return_dict=True
                )
            else:
                # assert all([x in transformer_inputs for x in ['input_ids', 'attention_mask', 'token_type_ids']])
                output_dict = self.transformer_model(**transformer_inputs, return_dict=True)
        
        # Dimension reduction
        if self.model_args.reduce_dim:
            tok_reps = self.tok_proj(output_dict.last_hidden_state) # bs, seq_len, tok_dim
        else:
            tok_reps = output_dict.last_hidden_state

        # Sense Disambiguation
        # # Sense Embeddings from static embedding layer
        tok_sense_embeds = self.sense_embeds[transformer_inputs['input_ids']] # bs, seq_len, num_senses, tok_dim
        # # Sense disambiguation using cross attention
        sdu_outputs = self.sdu(
            tok_reps, tok_sense_embeds,
            output_attentions=output_sense_attention_weights
        ) # sense disambiguated representation| bs, seq_len, tok_dim
        
        if self.model_args.add_positional_embeddings:
            e = self.add_pos_embeddings(sdu_outputs[0])
        else:
            e = sdu_outputs[0]

        # Spatial Gating Unit
        sgu_outputs = self.sgu(tok_reps, e, transformer_inputs['attention_mask'])
        
        emb = self.channel_proj(sgu_outputs[0])

        tok_dim = tok_reps.shape[-1]
        # Use only N 
        if sample_token_ind is not None:
            sampled_ind = sample_token_ind.unsqueeze(2).expand(sample_token_ind.shape[0], sample_token_ind.shape[1], tok_dim)
            sampled_emb = torch.gather(emb, 1, sampled_ind).view(-1, tok_dim) # bs*N, tok_dim
            sampled_tok_hs = torch.gather(tok_reps, 1, sampled_ind).view(-1, tok_dim) # bs*N, tok_dim
        else:
            sampled_emb = emb
            sampled_tok_hs = tok_reps

        return RandomSampleOutput(
            sample_token_emb = sampled_emb,
            transformer_sample_token_emb= sampled_tok_hs if output_transformer_embs else None,
            sense_attention_weights = sdu_outputs[1] if output_sense_attention_weights else None,
        )

    def embed(self, output_transformer_embs=False, **features):
        outputs = self.forward_representation(
            features, sample_token_ind=None, 
            output_transformer_embs=output_transformer_embs
        )
        return SeqOutput(
            token_emb = outputs.sample_token_emb,
            transformer_token_emb = outputs.transformer_sample_token_emb if output_transformer_embs else None,
        )
    
