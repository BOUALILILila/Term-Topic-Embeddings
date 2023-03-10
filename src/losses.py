# coding=utf-8
from typing import Tuple, Any, Optional, Union

import torch
from torch import nn

from transformers import (
    BatchEncoding,
    PreTrainedModel,
    DistilBertModel,
)
from transformers.utils import logging


from arguments import ModelArguments
from model_utils import (
    BaseModel, LossModel, 
    get_max_similarity
)

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

def mask_cls_sep(seq_mask):
    sep_ind = (seq_mask>0).count_nonzero(dim=1)-1 # the last non zero indices along batc dim
    sep_ind = (torch.arange(seq_mask.shape[0]), sep_ind)
    smask = seq_mask.clone()
    smask[sep_ind] = torch.zeros(1).to(seq_mask.device).to(seq_mask.dtype)
    smask[:,0] = torch.zeros(seq_mask.shape[0]).to(seq_mask.device).to(seq_mask.dtype)
    return smask

class EmbLossTransformerEmb(LossModel):
    def __init__(
        self, 
        model: BaseModel, 
        args: Optional[ModelArguments] = None,     
    ):
        super().__init__(model, args)
        if args.emb_loss is not None:
            if args.emb_loss == "cos":
                logger.info(">> Using Cosine embedding loss")
                self.loss = nn.CosineEmbeddingLoss(reduction="mean")
                self.target = True
            elif args.emb_loss == "mse":
                logger.info(">> Using MSE embedding loss")
                self.loss = nn.MSELoss()
                self.target = False
            else:
                raise ValueError(f"Unkown {args.emb_loss} embedding loss.")
        else:
            logger.info(">> Using MSE embedding loss")
            self.loss = nn.MSELoss()
            self.target = False
    
    def forward(
        self, 
        transformer_inputs: BatchEncoding, 
        sample_token_ind: Optional[torch.Tensor] = None, 
        **kwargs: Any
    ) -> Tuple[torch.tensor, Any]: 
        """
        Parameters
        ----------
        transformer_inputs : BatchEncoding
            Dict with input_ids and attention_mask.
        sample_token_ind : torch.Tensor of shape (bs,N)
            Indices of the sampled random tokens. 
            if None use all seq tokens.
        Returns
        -------
        loss: torch.tensor of shape (1,)
            The batch loss.
        """
        # Get embeddings
        if sample_token_ind is not None:
            out = self.model.forward_representation(
                transformer_inputs=transformer_inputs, 
                sample_token_ind=sample_token_ind, 
                output_transformer_embs=True,
                **kwargs
            )
            token_emb = out.sample_token_emb # (bs * N, dim)
            transformer_token_emb = out.transformer_sample_token_emb # (bs * N, dim)
        else:
            out = self.model.embed(output_transformer_embs=True, **transformer_inputs)
            seq_emb = out.token_emb # (bs, seq_length, dim)
            transformer_seq_emb = out.transformer_token_emb  # (bs, seq_length, dim)
            # mask select
            with torch.no_grad():
                mask = mask_cls_sep(transformer_inputs['attention_mask']) # mask cls and sep
            mask = mask.unsqueeze(-1).expand_as(seq_emb).bool()  # (bs, seq_length, dim)
            assert seq_emb.size() == transformer_seq_emb.size()
            dim = seq_emb.size(-1)

            token_emb = torch.masked_select(seq_emb, mask)  # (bs * seq_length * dim)
            token_emb = token_emb.view(-1, dim)  # (bs * seq_length, dim)

            transformer_token_emb = torch.masked_select(transformer_seq_emb, mask)
            transformer_token_emb = transformer_token_emb.view(-1, dim) # (bs * slct, dim)

        if self.target:
            target = token_emb.new(token_emb.size(0)).fill_(1)  # (bs * seq_length,) or (bs * N,)
            loss = self.loss(token_emb, transformer_token_emb, target)
        else:
            loss = self.loss(token_emb, transformer_token_emb)

        return (loss.unsqueeze(0), )

# Distillation and raking loss

class MarginMSELoss(LossModel):
    """
    Compute the MSE loss between the |sim(Query, Pos) - sim(Query, Neg)| and |gold_sim(Q, Pos) - gold_sim(Query, Neg)|.
    By default, sim() is the dot-product.
    For more details, please refer to https://arxiv.org/abs/2010.02666.
    """
    def __init__(
        self, 
        model: BaseModel, 
        query_model: Union[str, BaseModel, PreTrainedModel] = "same",
        args: Optional[ModelArguments] = None,
        ):

        super().__init__(model, args)
        self.model = model
        if query_model in ["same", "oracle"]:
            self.query_model_type = query_model
            self.query_model = self.model
        else:
            self.query_model_type = "pretrained_model"
            self.query_model = query_model

        logger.info(f'>> Query token representations are produced by {self.query_model_type}')

        logger.info('>> Similarity is computed on the token level')
        self.similarity_fct = get_max_similarity(self.args.similarity_fct)
        self.loss = nn.MSELoss()

    def forward(
        self, 
        query_inputs: BatchEncoding,
        pos_inputs: BatchEncoding,
        neg_inputs: BatchEncoding,
        labels: torch.Tensor,
        **kwargs
        ):

        pos_embs = self.model.embed(**pos_inputs).token_emb
        neg_embs = self.model.embed(**neg_inputs).token_emb

        if self.query_model_type == "pretrained_model":
            if isinstance(self.query_model, DistilBertModel):
                output_dict = self.query_model(
                    input_ids=query_inputs['input_ids'], attention_mask=query_inputs['attention_mask'], 
                    return_dict=True
                )
            else:
                output_dict = self.query_model(**query_inputs, return_dict=True) 
            query_embs = output_dict.last_hidden_state
        else:
            output = self.query_model.embed(**query_inputs, output_transformer_embs=True)
            query_embs = output.token_emb if self.query_model_type=="same" else output.transformer_token_emb

        # Mask CLS and SEP
        with torch.no_grad():
            query_mask = mask_cls_sep(query_inputs['attention_mask'])
            pos_mask = mask_cls_sep(pos_inputs['attention_mask'])
            neg_mask = mask_cls_sep(neg_inputs['attention_mask'])
        
        scores_pos = self.similarity_fct(query_embs, query_mask, pos_embs, pos_mask)
        scores_neg = self.similarity_fct(query_embs, query_mask, neg_embs, neg_mask)
        margin_pred = scores_pos - scores_neg

        loss = self.loss(margin_pred, labels)

        return (loss.unsqueeze(0),)

class MarginMSEWithEmbLoss(LossModel):
    """
    Compute the MSE loss between the |sim(Query, Pos) - sim(Query, Neg)| and |gold_sim(Q, Pos) - gold_sim(Query, Neg)|.
    Adds a loss on the embeddings so that the prouced embeddings stay close to the transformer embeddings.
    """
    def __init__(
        self, 
        model: BaseModel, 
        args: Optional[ModelArguments] = None,
        ):

        super().__init__(model, args)
        self.model = model

        self.similarity_fct = get_max_similarity(self.args.similarity_fct)
        
        self.loss = nn.MSELoss()
        self.alpha_mmse = args.alpha_mmse

        self.output_transformer_embs = True
        if args.emb_loss is None:
          self.output_transformer_embs = False
        elif args.emb_loss == "cos":
          logger.info(">> Using Cosine embedding loss")
          self.emb_loss = nn.CosineEmbeddingLoss(reduction="mean")
          self.target = True
        elif args.emb_loss == "mse":
          logger.info(">> Using MSE embedding loss")
          self.emb_loss = nn.MSELoss()
          self.target = False
        else:
          raise ValueError(f"Unkown {args.emb_loss} embedding loss.")
        self.alpha_emb_loss = args.alpha_emb_loss

    def forward(
        self, 
        query_inputs: BatchEncoding,
        pos_inputs: BatchEncoding,
        neg_inputs: BatchEncoding,
        labels: torch.Tensor,
        **kwargs
        ):

        q_outputs = self.model.embed(output_transformer_embs=self.output_transformer_embs, **query_inputs)
        query_embs = q_outputs.token_emb
        pos_outputs = self.model.embed(output_transformer_embs=self.output_transformer_embs, **pos_inputs)
        pos_embs = pos_outputs.token_emb
        neg_outputs = self.model.embed(output_transformer_embs=self.output_transformer_embs, **neg_inputs)
        neg_embs = neg_outputs.token_emb

        # Mask CLS and SEP
        with torch.no_grad():
            query_mask = mask_cls_sep(query_inputs['attention_mask'])
            pos_mask = mask_cls_sep(pos_inputs['attention_mask'])
            neg_mask = mask_cls_sep(neg_inputs['attention_mask'])

        # Ranking score loss
        scores_pos = self.similarity_fct(query_embs, query_mask, pos_embs, pos_mask)
        scores_neg = self.similarity_fct(query_embs, query_mask, neg_embs, neg_mask)
        margin_pred = scores_pos - scores_neg

        margin_mse_loss = self.loss(margin_pred, labels)
        loss = self.alpha_mmse * margin_mse_loss

        # Embedding loss
        if self.output_transformer_embs:
            embs = [
                (query_embs, q_outputs.transformer_token_emb, query_mask), 
                (pos_embs, pos_outputs.transformer_token_emb, pos_mask), 
                (neg_embs, neg_outputs.transformer_token_emb, neg_mask)
            ]
            all_s_hidden_states_slct, all_t_hidden_states_slct, all_targets = [], [], []
            
            for s_hidden_states, t_hidden_states, attention_mask in embs:
                mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states).bool()  # (bs, seq_length, dim)
                assert s_hidden_states.size() == t_hidden_states.size()
                dim = s_hidden_states.size(-1)

                s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
                s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
                all_s_hidden_states_slct.append(s_hidden_states_slct)
                t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
                t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
                all_t_hidden_states_slct.append(t_hidden_states_slct)
                
                if self.target:
                    target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
                    all_targets.append(target)

            if self.target:
                emb_loss = self.emb_loss(torch.cat(all_s_hidden_states_slct), torch.cat(all_t_hidden_states_slct), torch.cat(all_targets))
            else:
                emb_loss = self.emb_loss(torch.cat(all_s_hidden_states_slct), torch.cat(all_t_hidden_states_slct))
            
            loss += self.alpha_emb_loss * emb_loss

        return (loss.unsqueeze(0),)