# coding=utf-8
import os
from typing import Callable, Optional, Union
from dataclasses import dataclass
import copy

import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from transformers import (
    AutoModel,
    AutoConfig,
    WEIGHTS_NAME,
)
from transformers.activations import gelu
from transformers.utils import logging


logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

############################ Constants ############################
SENSE_EMBED_WEIGHTS_NAME = 'model.pt'
MODEL_ARGS_NAME = 'model_args.pt'

############################ Functions ############################
def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the btach cosine similarity cos_sim(a[i], b[i]) for all i.
    :return: Matrix with res[i]  = cos_sim(a[i], b[i])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 2:
        a = a.unsqueeze(1)

    if len(b.shape) == 2:
        b = b.unsqueeze(1)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=2)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=2)
    return torch.bmm(a_norm, b_norm.transpose(-1, -2))


def dot_score(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the btach dot-product dot_prod(a[i], b[i]) for all i.
    :return: Matrix with res[i]  = dot_prod(a[i], b[i])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 2:
        a = a.unsqueeze(1)

    if len(b.shape) == 2:
        b = b.unsqueeze(1)

    return torch.bmm(a, b.transpose(-1, -2))

SIMILARITY_FUNCTIONS = {
    'cos' : cos_sim,
    'dot' : dot_score,
}

def get_similarity_fct(tag: str) -> callable:
    if tag not in SIMILARITY_FUNCTIONS:
        raise ValueError(
            f"{tag} is not supported, only {', '.join(SIMILARITY_FUNCTIONS.keys())} are supported."
        )
    return SIMILARITY_FUNCTIONS[tag]


def max_dot_score(a: torch.Tensor, amask: torch.Tensor, b: torch.Tensor, bmask: torch.Tensor):
    """
    Computes the max dot-product dot_prod(a[i], b[i])
    :return: Vector with res[i] = max(dot_prod(a[i], b[i]), dim=1)
    """
    assert len(a.shape) == 3 and len(b.shape) == 3 , f"Pairwise_dot_score on a.shape = {a.shape} and b.shape = {b.shape}. a and b must have a shape of (bs, seqlen, D)." 
    # Masking
    a = a * amask.unsqueeze(-1)
    b = b * bmask.unsqueeze(-1)

    # Dot product
    score = torch.bmm(a, b.transpose(2,1))
    
    # mask out padding on the doc dimension (mask by -1000, because max should not select those, setting it to 0 might select them)
    exp_mask = bmask.bool().unsqueeze(1).expand(-1, score.shape[1],-1)
    score[~exp_mask] = -10000

    # max pooling over document dimension
    score = score.max(-1).values

    # mask out paddding query values
    score[~(amask.bool())] = 0

    # sum over query values
    score = score.sum(-1)

    return score

def max_cos_sim(a: torch.Tensor, amask: torch.Tensor, b: torch.Tensor, bmask: torch.Tensor):
    """
    Computes the max cossim cos_sim(a[i], b[i])
    :return: Vector with res[i] = max(cos_sim(a[i], b[i]), dim=1)
    """
    # Normalize
    anorm = torch.nn.functional.normalize(a, p=2, dim=2)
    bnorm = torch.nn.functional.normalize(b, p=2, dim=2)
    
    return max_dot_score(anorm, amask, bnorm, bmask)

MAX_SIMILARITY_FUNCTIONS = {
    'cos' : max_cos_sim,
    'dot' : max_dot_score,
}

def get_max_similarity(tag: str) -> callable:
    if tag not in MAX_SIMILARITY_FUNCTIONS:
        raise ValueError(
            f"{tag} is not supported, only {', '.join(MAX_SIMILARITY_FUNCTIONS.keys())} are supported."
        )
    return MAX_SIMILARITY_FUNCTIONS[tag]


def pairwise_dot_score(a: torch.Tensor, amask: torch.Tensor, b: torch.Tensor, bmask: torch.Tensor):
    """
   Computes the pairwise dot-product dot_prod(a[i], b[i])
   :return: Vector with res[i] = dot_prod(a[i], b[i])
   """
    assert len(a.shape) == 2 and len(b.shape) == 2 , f"Pairwise_score on a.shape = {a.shape} and b.shape = {b.shape}. a and b must be two-dimensional (bs,D)" 
    
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    return (a * b).sum(dim=-1)


def pairwise_cos_sim(a: torch.Tensor, amask: torch.Tensor, b: torch.Tensor, bmask: torch.Tensor):
    """
   Computes the pairwise cossim cos_sim(a[i], b[i])
   :return: Vector with res[i] = cos_sim(a[i], b[i])
   """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    return pairwise_dot_score(
        torch.nn.functional.normalize(a, p=2, dim=1), 
        amask,
        torch.nn.functional.normalize(b, p=2, dim=1),
        bmask
    )

PAIRWISE_SIMILARITY_FUNCTIONS = {
    'cos' : pairwise_cos_sim,
    'dot' : pairwise_dot_score,
}

def get_pairwise_similarity(tag:str) -> callable:
    if tag not in PAIRWISE_SIMILARITY_FUNCTIONS:
        raise ValueError(
            f"{tag} is not supported, only {', '.join(PAIRWISE_SIMILARITY_FUNCTIONS.keys())} are supported."
        )
    return PAIRWISE_SIMILARITY_FUNCTIONS[tag]

############################ Models ############################

@dataclass
class RandomSampleOutput:
    sample_token_emb: torch.FloatTensor = None
    transformer_sample_token_emb: torch.FloatTensor = None
    sense_attention_weights: torch.FloatTensor = None
    word_mask: torch.Tensor = None

@dataclass
class SeqOutput:
    token_emb: torch.FloatTensor = None
    transformer_token_emb: torch.FloatTensor = None
    word_mask: torch.Tensor = None



class BaseModel(nn.Module):

    @classmethod
    def from_pretrained(
            cls, *args, **kwargs
    ):
        """ Loads a model saved on local storage.
        """
        model_args = kwargs.pop('model_args', None) 
        freeze_transformer_weights = kwargs.pop('freeze_transformer_weights', False) 
        # sense_embed_init = model_args.sense_embed_init

        path = args[0]

        if model_args is None and os.path.exists(os.path.join(path, MODEL_ARGS_NAME)): 
            logger.info(f"Loading model arguments from {os.path.join(path, MODEL_ARGS_NAME)}")
            model_args = torch.load(os.path.join(path, MODEL_ARGS_NAME))
        if model_args is None:
            logger.error(f"The parameter model_args was not specified and {path} does not have a {MODEL_ARGS_NAME} file.")
        
        # Load core model: contextualized transformer model
        MODEL_CLS = AutoModel
        if hasattr(model_args,'positionless_core'): # compat with old ckpts
            if model_args.positionless_core:
                logger.info(f"Discarding position embeddings for contextualized transformer model core...")
                MODEL_CLS = PositionlessDistilBert
                
        hf_model = MODEL_CLS.from_pretrained(*args, **kwargs)

        # Load model
        model = cls(hf_model, model_args, freeze_transformer_weights=freeze_transformer_weights)
        
        if os.path.exists(os.path.join(path, SENSE_EMBED_WEIGHTS_NAME)):
            logger.info(f'Loading embedding and attention layer weights from {os.path.join(path, SENSE_EMBED_WEIGHTS_NAME)}')
            model_dict = torch.load(os.path.join(path, SENSE_EMBED_WEIGHTS_NAME), map_location="cpu")
            load_result = model.load_state_dict(model_dict, strict=False)
            if len(load_result.missing_keys) == 0:
                logger.error(f"The transformer model keys should be missing when loading checkpoing. These keys were loaded separately.")
            elif not all(key.startswith('transformer_model') for key in set(load_result.missing_keys)):
                logger.error(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}")
            if len(load_result.unexpected_keys) != 0:
                logger.warn(f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}.")

        return model

    def save_pretrained(self, output_dir: str, save_function: Callable = torch.save):
        """ Saves the model on local storage.
        from: https://github.com/luyug/COIL/blob/main/modeling.py 
        """
        self.transformer_model.save_pretrained(output_dir, save_function = save_function)
        model_dict = self.state_dict()
        # remove the transformer model parameters already saved
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('transformer_model')]
        for k in hf_weight_keys:
            model_dict.pop(k)
        
        save_function(model_dict, os.path.join(output_dir, SENSE_EMBED_WEIGHTS_NAME))
        save_function(self.model_args, os.path.join(output_dir, MODEL_ARGS_NAME))

    def load(self, path: str):
        # We load the model state dict on the CPU to avoid an OOM error.
        transformer_state_dict = torch.load(os.path.join(path, WEIGHTS_NAME), map_location="cpu")
        logger.info(f'Loading embedding and attention layer weights from {os.path.join(path, SENSE_EMBED_WEIGHTS_NAME)}')
        sb_model_state_dict = torch.load(os.path.join(path, SENSE_EMBED_WEIGHTS_NAME), map_location="cpu")
        # If the model is on the GPU, it still works!
        
        load_result = self.transformer_model.load_state_dict(transformer_state_dict, strict=False)

        if len(load_result.missing_keys) != 0:
            if set(load_result.missing_keys) == set(self.transformer_model._keys_to_ignore_on_save):
                self.transformer_model.tie_weights()
            else:
                logger.warn(f"There were missing keys in the transformer checkpoint model loaded: {load_result.missing_keys}.")
        if len(load_result.unexpected_keys) != 0:
            logger.warn(f"There were unexpected keys in the transformer checkpoint model loaded: {load_result.unexpected_keys}.")
        
        load_result = self.load_state_dict(sb_model_state_dict, strict=False)

        if len(load_result.missing_keys) == 0:
            logger.error(f"The transformer model keys should be missing when loading checkpoing. These keys were loaded separately.")
        elif not all(key.startswith('transformer_model') for key in set(load_result.missing_keys)):
            logger.error(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}")
        if len(load_result.unexpected_keys) != 0:
            logger.warn(f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}.")
        return load_result

    def init_weights(self, module):
        module.apply(self.transformer_model._init_weights)
        # can customize to other initializing functions
    
    def init_embedding(self, vocab_size, num_senses, token_dim):
        layers = [torch.zeros((vocab_size, token_dim)).float() for _ in range(num_senses)]
        for layer in layers:
            torch.nn.init.xavier_uniform_(layer)

        stacked = torch.stack(layers, dim=1)
        assert stacked.shape == (vocab_size, num_senses, token_dim)
        return stacked


class LossModel(nn.Module):
    def __init__(
        self, 
        model: BaseModel, 
        args = None,     
    ):
        super().__init__()
        self.model = model
        self.args = args

    def save_pretrained(self, output_dir: str, save_function: Callable = torch.save):
        self.model.save_pretrained(output_dir, save_function)

import math

# Modified from: https://github.com/huggingface/transformers/blob/v4.9.2/src/transformers/models/bert/modeling_bert.py#L227
# Discard the value layer and the dropout
class MyBertSelfAttention(nn.Module):
    def __init__(self, config, pool_raw_scores=False, query_input_dim=None, temperature=1.):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        query_input_hidden_size = query_input_dim if query_input_dim is not None else config.hidden_size
        self.query = nn.Linear(query_input_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        # self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

        self.pool_raw_scores = pool_raw_scores
        self.temperature = temperature

        logger.info(f"Using softmax temperature = {self.temperature}")

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        ):

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            # value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            # value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            # value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            # value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            # value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # if self.is_decoder:
        #     # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        #     # Further calls to cross_attention layer can then reuse all cross-attention
        #     # key/value_states (first "if" case)
        #     # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        #     # all previous decoder key/value_states. Further calls to uni-directional self-attention
        #     # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        #     # if encoder bi-directional self-attention `past_key_value` is always `None`
        #     past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        if self.pool_raw_scores:
            attention_scores = torch.max(attention_scores, dim=1)[0]
            attention_probs = nn.Softmax(dim=-1)(attention_scores/self.temperature)
        else:
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        # don't need the new rep
        context_layer = None

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=True,
        ):
        """
            returns the attention weights of shape hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[1]
            if self attention
            else (cross attention)
                output shape = hidden_states.shape[0], hidden_states.shape[1], encoer_hidden_states.shape[1]
        """
        attn_outputs =  self._forward(hidden_states, attention_mask, head_mask, 
                        encoder_hidden_states, encoder_attention_mask,
                        past_key_value, output_attentions)
        
        # if pool_raw_scores=False attn_probs are in attn_outputs[1] : shape = B, num_attn_heads, X_seq_len, X_seq_len | Y_seq_len
        # else : attn_probs are in attn_outputs[1] : shape = B, X_seq_len, X_seq_len | Y_seq_len
        # Max Pool each sense dimension
        if self.pool_raw_scores:
            return attn_outputs[1]
        else:
            max_scores = torch.max(attn_outputs[1], dim=1)[0]
            return nn.Softmax(dim=-1)(max_scores/self.temperature) # B, X_seq_len, Y_seq_len


class SlidingWindowAttenion(MyBertSelfAttention):
    def __init__(self, config, context_window: int):
        """
            context_window refers to the number of positions to the left|right of the central token!!
                not the whole window(2*context_window+1)
        """
        super().__init__(config)
        if context_window < config.max_position_embeddings :
            diags = [
                torch.diag(torch.ones(config.max_position_embeddings - abs(i), dtype=int),-i,) 
                for i in range(-context_window, context_window+1)
                if abs(i) < config.max_position_embeddings
            ]
            self.sliding_mask = torch.stack(diags).sum(0)
        else:
            self.sliding_mask = torch.ones(config.max_position_embeddings,config.max_position_embeddings, dtype=int)

        self.sliding_mask.requires_grad = False
        
    def _get_attn_mask(self, seq_len: int, dtype, DEVICE, seq_mask:torch.Tensor = None):
        sm = self.sliding_mask[None, None, :seq_len, :seq_len].to(DEVICE) # restrict seq length to the actual batch sequence length
        
        # Mask CLS and SEP
        sep_ind = (seq_mask>0).count_nonzero(dim=1)-1 # the last non zero indices along batc dim
        sep_ind = (torch.arange(seq_mask.shape[0]), sep_ind)
        smask = seq_mask.clone()
        smask[sep_ind] = torch.zeros(1).to(seq_mask.device).to(seq_mask.dtype)
        smask[:,0] = torch.zeros(seq_mask.shape[0]).to(seq_mask.device).to(seq_mask.dtype)
        
        if seq_mask is not None:
            sm = smask[:, None, None, :] * sm.to(seq_mask.dtype)
        sm = sm.to(dtype=dtype)  # fp16 compatibility
        return (1.0 - sm) * -10000.0
    
    def _forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=True,
        ):

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            # value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            # value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            # value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            # value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            # value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # if self.is_decoder:
        #     # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        #     # Further calls to cross_attention layer can then reuse all cross-attention
        #     # key/value_states (first "if" case)
        #     # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        #     # all previous decoder key/value_states. Further calls to uni-directional self-attention
        #     # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        #     # if encoder bi-directional self-attention `past_key_value` is always `None`
        #     past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        max_pool_on_heads = torch.max(attention_scores, dim=1)[0]
        attention_probs = nn.Softmax(dim=-1)(max_pool_on_heads) # B, X_seq_len, X_seq_len

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        # don't need the new rep
        context_layer = None

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=True,
        ):
        """
            returns the attention weights of shape hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[1]
            if self attention
            else (cross attention)
                output shape = hidden_states.shape[0], hidden_states.shape[1], encoer_hidden_states.shape[1]
        """
        with torch.no_grad():
            sliding_attn_mask = self._get_attn_mask(
                hidden_states.shape[1], hidden_states.dtype, hidden_states.device, 
                attention_mask) # B, 1, X_seq_len, X_seq_len

        attn_outputs =  self._forward(hidden_states, sliding_attn_mask, head_mask, 
                        encoder_hidden_states, encoder_attention_mask,
                        past_key_value, output_attentions)
        
        # attn_probs are in attn_outputs[1] : shape = B, X_seq_len, X_seq_len 
        return attn_outputs[1]

class SenseDisambiguationUnit(nn.Module):
    def __init__(
        self, 
        pool_raw_scores: bool=False,
        num_attn_heads: Optional[int]=None,
        dropout_prob: Optional[float]=0.,
        hidden_size: Optional[int]=None,
        config: Optional[AutoConfig]=None, 
        query_input_dim: Optional[int]=None,
        temperature: Optional[float]=1.,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else AutoConfig.from_pretrained("bert-base-uncased")
        if num_attn_heads is not None:
            self.config.num_attention_heads = num_attn_heads
        self.config.is_decoder = True # used by BertAttention -> BertSelfAttention
        self.config.add_cross_attention = True # Just in case
        self.config.attention_probs_dropout_prob = dropout_prob
        if hidden_size is not None:
            self.config.hidden_size = hidden_size  
        self.config.output_attentions = True
        
        self.attn = MyBertSelfAttention(self.config, pool_raw_scores, query_input_dim, temperature)
        self.apply(self._init_weights)
    
    def forward(
        self,
        tok_reps, # contextualized reps (bs, seq_len, tok_dim)
        tok_sense_embeds, # static sense embeddings (bs, seq_len, num_senses, tok_dim)
        tok_attention_mask=None,
        sense_attention_mask=None,
        output_attentions=False,
    ):  
        bs = tok_reps.shape[0]
        num_senses = tok_sense_embeds.shape[-2]
        tok_sense_embeds = tok_sense_embeds.view(-1, num_senses, self.config.hidden_size) 

        sense_weights = self.attn( # Attend(X,Y) word attend to senses 
            tok_reps.view(-1, 1, tok_reps.shape[-1]), # X hidden states| bs*seq_len, 1, tok_dim
            attention_mask = tok_attention_mask,
            encoder_hidden_states = tok_sense_embeds, # Y hidden states
            encoder_attention_mask = sense_attention_mask,
            output_attentions = True,
        ) # bs*seq_len, 1, num_senses
        se = torch.matmul(sense_weights, tok_sense_embeds).view(bs,-1,self.config.hidden_size)
        
        outputs = (se, sense_weights, ) if output_attentions else (se, )
        return outputs
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

class SlidingWindowSelfAttenion(nn.Module):
    def __init__(self, config, context_window, use_value=False, query_input_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        query_input_hidden_size = query_input_dim if query_input_dim is not None else config.hidden_size
        self.query = nn.Linear(query_input_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.use_value = use_value
        if self.use_value:
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
        ####

        if context_window < config.max_position_embeddings :
            diags = [
                torch.diag(torch.ones(config.max_position_embeddings - abs(i), dtype=int),-i,) 
                for i in range(-context_window, context_window+1)
                if abs(i) < config.max_position_embeddings
            ]
            self.sliding_mask = torch.stack(diags).sum(0)
        else:
            self.sliding_mask = torch.ones(config.max_position_embeddings,config.max_position_embeddings, dtype=int)

        self.sliding_mask.requires_grad = False

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def _get_attn_mask(self, seq_len: int, dtype, DEVICE, seq_mask:torch.Tensor = None):
        sm = self.sliding_mask[None, None, :seq_len, :seq_len].to(DEVICE) # restrict seq length to the actual batch sequence length
        
        if seq_mask is not None:
            # Mask CLS and SEP
            sep_ind = (seq_mask>0).count_nonzero(dim=1)-1 # the last non zero indices along batc dim
            sep_ind = (torch.arange(seq_mask.shape[0]), sep_ind)
            smask = seq_mask.clone()
            smask[sep_ind] = torch.zeros(1).to(seq_mask.device).to(seq_mask.dtype)
            smask[:,0] = torch.zeros(seq_mask.shape[0]).to(seq_mask.device).to(seq_mask.dtype)
            sm = smask[:, None, None, :] * sm.to(seq_mask.dtype)
        
        sm = sm.to(dtype=dtype)  # fp16 compatibility
        return (1.0 - sm) * -10000.0
    
    def _forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=True,
        ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            if self.use_value:
                value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            if self.use_value:
                value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            if self.use_value:
                value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            if self.use_value:
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            if self.use_value:
                value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            if self.use_value:
                past_key_value = (key_layer, value_layer)
            else:
                past_key_value = (key_layer,)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        if not self.use_value:
            attention_scores = torch.max(attention_scores, dim=1)[0]

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        if self.use_value:
            context_layer = torch.matmul(attention_probs, value_layer)

            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
        else:
            context_layer = None

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=True,
        ):
        """
            returns the attention weights of shape hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[1]
            if self attention
            else (cross attention)
                output shape = hidden_states.shape[0], hidden_states.shape[1], encoer_hidden_states.shape[1]
        """
        is_cross = encoder_hidden_states is not None
        
        with torch.no_grad():
            if is_cross: # cross attn
                attention_mask = encoder_attention_mask
            sliding_attn_mask = self._get_attn_mask(
                hidden_states.shape[1], hidden_states.dtype, hidden_states.device, 
                attention_mask) # B, 1, X_seq_len, X_seq_len
        
        if not output_attentions and not self.use_value:
            # logger.warn("Need to output Value and/or Attention probs but none of them were set...\n >> Forcing output_attentions to True.")
            output_attentions = True

        # is is_cross use the encoder_sliding_attn_mask and the first slifinf_attn_mask is ignored and the other way around if selfattn
        attn_outputs =  self._forward(hidden_states, sliding_attn_mask, head_mask, 
                        encoder_hidden_states, sliding_attn_mask,
                        past_key_value, output_attentions)
        
        return attn_outputs

class AttnSpatialGatingUnit(nn.Module):
    def __init__(
        self, 
        sliding_cntxt_window: int,
        num_attn_heads: Optional[int]=None,
        dropout_prob: Optional[float]=0.,
        hidden_size: Optional[int]=None,
        config: Optional[AutoConfig]=None, 
        use_value: Optional[bool]=False,
        query_input_dim: Optional[int]=None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else AutoConfig.from_pretrained("bert-base-uncased")
        if num_attn_heads is not None:
            self.config.num_attention_heads = num_attn_heads
        self.config.attention_probs_dropout_prob = dropout_prob
        if hidden_size is not None:
            self.config.hidden_size = hidden_size  
        self.config.hidden_size = self.config.hidden_size//2 # the attention is computed over half the hidden dimension
        self.config.output_attentions = True

        self.attn = SlidingWindowSelfAttenion(
            self.config, sliding_cntxt_window, 
            use_value=use_value, query_input_dim=query_input_dim
        )
        self.norm = nn.LayerNorm([self.config.hidden_size])
        self.apply(self._init_weights)
    
    def forward(
        self,
        tok_reps, # token reps (bs, seq_len, tok_dim)
        tok_attention_mask=None, # seq attention mask (bs, seq_len)
        output_attentions=False,
    ):  
        res, gate = tok_reps.chunk(2, dim=-1)
        gate = self.norm(gate)

        attn_outputs = self.attn( # Attend(X,X-cntxt) word attend to its cntxt 
            gate, # bs, seq_len, tok_dim/2
            attention_mask = tok_attention_mask,
            output_attentions = output_attentions,
        ) # bs, seq_len, tok_dim/2 | bs, seq_len, seq_len tuple

        gate = attn_outputs[0] if attn_outputs[0] is not None else torch.matmul(attn_outputs[1], gate) # bs, seq_len, tok_dim/2

        res = torch.cat([res,gate], dim=-1)  # bs, seq_len, tok_dim

        outputs = (res, attn_outputs[1], ) if output_attentions else (res, )
        return outputs
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class AttnSpatialGatingUnitFromTransformer(nn.Module):
    """ 
    Query: from the transformer representations
    Key: from the transformer representation
    self-attention scores from the transformer used for weighting the gate vectors
    """
    def __init__(
        self, 
        sliding_cntxt_window: int,
        num_attn_heads: Optional[int]=None,
        dropout_prob: Optional[float]=0.,
        hidden_size: Optional[int]=None,
        config: Optional[AutoConfig]=None, 
        use_value: Optional[bool]=False,
        query_input_dim: Optional[int]=None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else AutoConfig.from_pretrained("bert-base-uncased")
        if num_attn_heads is not None:
            self.config.num_attention_heads = num_attn_heads
        self.config.attention_probs_dropout_prob = dropout_prob
        if hidden_size is not None:
            self.config.hidden_size = hidden_size  
        self.config.hidden_size = self.config.hidden_size
        self.config.output_attentions = True

        assert use_value==False ,"AttnSpatialGatingUnitFromTransformer can't use the value layer."

        self.attn = SlidingWindowSelfAttenion(
            self.config, sliding_cntxt_window, 
            use_value=use_value, query_input_dim=query_input_dim
        )
        self.norm = nn.LayerNorm([self.config.hidden_size//2]) # for the gate
        self.apply(self._init_weights)
    
    def forward(
        self,
        transformer_tok_reps, # token reps (bs, seq_len, tok_dim)
        se_tok_reps,
        tok_attention_mask=None, # seq attention mask (bs, seq_len)
        output_attentions=False,
    ):  
        res, gate = se_tok_reps.chunk(2, dim=-1)
        gate = self.norm(gate)

        attn_outputs = self.attn( # Attend(X,X-cntxt) word attend to its cntxt 
            hidden_states=transformer_tok_reps, # bs, seq_len, tok_dim
            attention_mask=tok_attention_mask,
            output_attentions=output_attentions,
        ) # bs, seq_len, tok_dim | bs, seq_len, seq_len tuple

        gate = torch.matmul(attn_outputs[1], gate) # bs, seq_len, tok_dim/2

        res = torch.cat([res,gate], dim=-1)  # bs, seq_len, tok_dim

        outputs = (res, attn_outputs[1], ) if output_attentions else (res, )
        return outputs
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class CrossAttnSpatialGatingUnit(AttnSpatialGatingUnit):
    """ Query: from the transformer representations
        Key,Value: from the SE representation
        cross-attention
    """
    
    def forward(
        self,
        transformer_tok_reps, # token reps (bs, seq_len, tok_dim)
        se_tok_reps,
        tok_attention_mask=None, # seq attention mask (bs, seq_len)
        output_attentions=False,
    ):  
        res, gate = se_tok_reps.chunk(2, dim=-1)
        gate = self.norm(gate)

        attn_outputs = self.attn( # Attend(X,X-cntxt) word attend to its cntxt 
            hidden_states=transformer_tok_reps, # bs, seq_len, tok_dim
            attention_mask=tok_attention_mask,
            encoder_hidden_states=gate, # bs, seq_len, tok_dim/2
            encoder_attention_mask=tok_attention_mask,
            output_attentions=output_attentions,
        ) # bs, seq_len, tok_dim/2 | bs, seq_len, seq_len tuple

        gate = attn_outputs[0] if attn_outputs[0] is not None else torch.matmul(attn_outputs[1], gate) # bs, seq_len, tok_dim/2

        res = torch.cat([res,gate], dim=-1)  # bs, seq_len, tok_dim

        outputs = (res, attn_outputs[1], ) if output_attentions else (res, )
        return outputs


##########

class PositionlessEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        # if config.sinusoidal_pos_embds:

        #     if is_deepspeed_zero3_enabled():
        #         import deepspeed

        #         with deepspeed.zero.GatheredParameters(self.position_embeddings.weight, modifier_rank=0):
        #             if torch.distributed.get_rank() == 0:
        #                 create_sinusoidal_embeddings(
        #                     n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight
        #                 )
        #     else:
        #         create_sinusoidal_embeddings(
        #             n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight
        #         )

        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        """
        Parameters:
            input_ids: torch.tensor(bs, max_seq_length) The token ids to embed.
        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        seq_length = input_ids.size(1)
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
        # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        word_embeddings = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        # position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = word_embeddings #+ position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings

from transformers.models.distilbert.modeling_distilbert import Transformer, DistilBertModel

class PositionlessDistilBert(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = PositionlessEmbeddings(config)  # Embeddings
        self.transformer = Transformer(config)  # Encoder

        self.init_weights()


# Modified from: https://github.com/huggingface/transformers/blob/v4.9.2/src/transformers/models/distilbert/modeling_distilbert.py
class EmbeddingLayerWithPoisition(nn.Module):
    def __init__(self, config, init=None):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        if init is not None:
            self.position_embeddings = copy.deepcopy(init)

        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_embeddings):
        seq_length = input_embeddings.shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embeddings.device)  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand(input_embeddings.shape[0],input_embeddings.shape[1])  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = input_embeddings + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings