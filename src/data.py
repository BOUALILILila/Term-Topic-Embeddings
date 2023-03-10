# coding=utf-8
from arguments import DataArguments
import os
from typing import (
  Optional, 
  Dict, 
  List, 
  Union,
)
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pickle
import tarfile, gzip
import json
import tqdm

import torch
from transformers import AutoTokenizer, BatchEncoding
from transformers.utils import logging

from data_utils import keep_vocab_item, http_get, WholeWordTokenizer

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

# Text collection iterator
class BaseDatasetIterator:
  def __iter__(self):
    raise NotImplementedError

  def load_dataset(self):
    raise NotImplementedError

class MsMarcoIterator(BaseDatasetIterator):

  def __init__(self, data_path: str) -> None:
    # self.nlp = sp.load("en_core_web_lg", disable=['parser', 'tagger', 'ner'])
    if not os.path.exists(data_path):
      raise ValueError(f"File not found at: {data_path}")
    self.data_path = data_path

  def __iter__(self):
    with open(self.data_path) as inF:
      for line in inF:
        pid, passage = line.rstrip().split('\t')
        yield passage
  
  def load_dataset(self):
    collection = defaultdict()
    with open(self.data_path) as inF:
      for line in inF:
        pid, passage = line.rstrip().split('\t')
        collection[pid] = passage
    return collection

# Collators
@dataclass
class RandomTokenCollator:
    """
    Data collator that will dynamically pad the inputs received.
    """
    tokenizer: AutoTokenizer
    max_seq_len: int = 128

    def __call__(
            self, features
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:

      transformer_inputs = [f['transformer_inputs'] for f in features]
      transformer_inputs = self.tokenizer.pad(
          transformer_inputs,
          padding='longest',
          max_length=self.max_seq_len,
          return_tensors='pt'
      )

      sample_token_ind = torch.tensor([f['sample_token_ind'] for f in features])

      return {
        'transformer_inputs': transformer_inputs, 
        'sample_token_ind': sample_token_ind,
        }

@dataclass
class SequenceTokenCollator:
    """
    Data collator that will dynamically pad the inputs received.
    """
    tokenizer: AutoTokenizer
    max_seq_len: int = 128

    def __call__(
            self, features
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:

      transformer_inputs = [f['transformer_inputs'] for f in features]
      transformer_inputs = self.tokenizer.pad(
          transformer_inputs,
          padding='longest',
          max_length=self.max_seq_len,
          return_tensors='pt'
      )

      return {
        'transformer_inputs': transformer_inputs, 
        }

# Samplers
class ShuffleTrainSampler:
  # For https://github.com/huggingface/transformers/blob/7d2feb3a3bedd0878039469260c98dd2d69c1be6/src/transformers/trainer_pt_utils.py#L667
  def set_epoch(self, epoch):
    self.rng = np.random.default_rng(self.seed + epoch)

  def __iter__(self):
    return iter(self.generate_samples())

class BaseTrainSamplesGenerator:
  def __init__(
    self, corpus_iter: BaseDatasetIterator, 
    tokenizer: AutoTokenizer,
    args: DataArguments,
    ):
    self.corpus_iter = corpus_iter
    self.tokenizer = tokenizer
    self.seed = args.rng_seed
    self.rng = np.random.default_rng(self.seed)
    self.max_seq_len = args.max_seq_len
    self.min_seq_len = args.min_seq_len
  
  def __len__(self):
    return self.corpus_count

  def generate_samples(self):
    collection = self.corpus_iter.load_dataset()
    all_ids = sorted(collection)
    if len(all_ids) == 0:
      raise RuntimeError("TrainDataset has no valid ids")
    while True:
      self.rng.shuffle(all_ids)
      for pid in all_ids:
        instance = self._create_single_instance(pid, collection[pid])
        if instance:
          yield instance.to_features()   
  
  def _create_single_instance(
    self, 
    i: int,
    passage: str,
    ):

    transformer_inputs = self.tokenizer.encode_plus(
      passage,
      max_length = self.max_seq_len,
      truncation= True,
      return_attention_mask = True,
      return_token_type_ids = True,
    )

    out = self._sample(transformer_inputs['input_ids'][1:-1])
    if self.valid(out) :
      return self.InputSample(str(i), transformer_inputs, **out)
    return None

  def valid(self, elements: dict):
    raise NotImplementedError
  
  def _sample(
    self,
    token_ids: List[int], 
    ):
    raise NotImplementedError

class SequenceTokensTrainSampler(BaseTrainSamplesGenerator, ShuffleTrainSampler, torch.utils.data.IterableDataset):
  def __init__(
    self, corpus_iter: BaseDatasetIterator, 
    tokenizer: AutoTokenizer,
    args: DataArguments,
    ):
    super().__init__(
      corpus_iter, tokenizer, args
      )
    if args.N is not None:
      logger.info("Using all sequence tokens during training (sampling strategy = 'none'), N is specified and will be ignored.")
    self.corpus_count = sum(1 for _ in self.corpus_iter)
  
  class InputSample(object):
    """ A sinlge Train instance """
    def __init__(self, 
      guid: str, 
      transformer_inputs: BatchEncoding,
      valid: bool,
      ) -> None:

      self.guid = guid
      self.transformer_inputs = transformer_inputs
    
    def to_features(self):
      features = {
        "transformer_inputs": self.transformer_inputs,
      }
      return features

  def valid(self, elements: dict):
    return elements['valid']
  
  def _sample(
    self,
    token_ids: List[int], 
    )-> Dict[str, bool]:
    if len(token_ids)> self.min_seq_len:
      return {'valid': True}
    return {'valid': False}

class RandomTokensTrainSampler(BaseTrainSamplesGenerator, ShuffleTrainSampler, torch.utils.data.IterableDataset):
  def __init__(
    self, corpus_iter: BaseDatasetIterator, 
    tokenizer: AutoTokenizer,
    args: DataArguments,
    ):
    super().__init__(
      corpus_iter, tokenizer, args
      )
    self.N = args.N
    assert self.N > 0 , "Number of random tokens per text must be greater than 0"
  
  class InputSample(object):
    """ A sinlge Train instance """
    def __init__(self, 
      guid: str, 
      transformer_inputs: BatchEncoding,
      token_ind: List[int],
      ) -> None:

      self.guid = guid
      self.transformer_inputs = transformer_inputs
      self.token_ind = token_ind
    
    def to_features(self):
      features = {
        "transformer_inputs": self.transformer_inputs,
        "sample_token_ind": self.token_ind, # N
      }
      return features

  def valid(self, elements: dict):
    return (len(elements['token_ind'])>0)


class FrqSubSamplingVocab:
  def __init__(
    self, corpus_iter: BaseDatasetIterator, 
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    min_count:int,
    sample:float,
    trim_rule:Optional[callable],
    train_path:str,
    ) -> None:
      VOCAB_KEYS = ['max_seq_len', 'min_count', 'sample']
      assert max_seq_len > 0 , "Max_seq_len must be greater than 0"
      params = locals()
      args_str = f'_{tokenizer.name_or_path.replace("/","-")}-' + '-'.join([f'{k}{params[k]}' for k in VOCAB_KEYS])
      self.vocab_path = os.path.splitext(train_path)[0] + args_str + '.pickle'
      self.load_vocab()
      if self.vocab is None:
        logger.info("Building vocabulary from tainining files ...")
        self.build_vocab(corpus_iter, tokenizer, max_seq_len, min_count, sample, trim_rule)
      self.save_vocab()

  def get_vocab(self):
    return self.vocab
  
  def get_corpus_count(self):
    return self.corpus_count

  def save_vocab(self):
    if self.vocab is not None and not os.path.exists(self.vocab_path):
      to_save = {'vocab': self.vocab, 'corpus_count': self.corpus_count}
      with open(self.vocab_path, 'wb') as fp:
        pickle.dump(to_save, fp, protocol=pickle.HIGHEST_PROTOCOL)
  
  def load_vocab(self):
    self.vocab = None
    self.corpus_count = 0
    if os.path.exists(self.vocab_path):
      logger.info(f"Loading vocabulary from saved file {self.vocab_path}")
      with open(self.vocab_path, 'rb') as fp:
        data = pickle.load(fp)
      self.vocab = data['vocab']
      self.corpus_count = data['corpus_count']

  def build_vocab(
    self, 
    corpus_iter: BaseDatasetIterator,
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    min_count: int,
    sample: float,
    trim_rule: Optional[Callable],
    ):
    raw_vocab, self.corpus_count = self._build_vocab(corpus_iter, tokenizer, max_seq_len)
    self.vocab = self._prep_vocab(raw_vocab, min_count, trim_rule, sample)
    self.vocab_size = len(self.vocab)

    # self.stats['raw_vocab_size'] = len(raw_vocab)
    # self.stats['vocab_size'] = self.vocab_size
    
  def _build_vocab(
    self, 
    corpus_iter: BaseDatasetIterator,
    tokenizer: AutoTokenizer, 
    max_seq_len: int,
    progress_per :str = 1000
    ) -> Tuple[int, int]:
    """Creates the vocab with their counts"""

    total_tokens = 0
    vocab = defaultdict(int)

    for i, passage in enumerate(corpus_iter):
      passage_ids = tokenizer.encode(
        passage,
        max_length = max_seq_len-2, # consider only seq_len or all passage tokens
        truncation = True,
        add_special_tokens=False,
      )
      if i % progress_per == 0:
        logger.info(
          f"PROGRESS: at passage #{i}, processed {total_tokens} tokens, {len(vocab)} unique tokens"
        )
      
      for token_id in passage_ids:
        vocab[token_id] += 1
      total_tokens += len(passage_ids)

    corpus_count = i + 1
    return vocab, corpus_count

  def _prep_vocab(
    self, 
    raw_vocab: Dict[str,int],
    min_count: int,
    trim_rule: Optional[Callable], 
    sample: float,
    ) -> Dict[str, Tuple[int,float]]:

    drop_total = drop_unique = 0
    retain_total, retain_tokens = 0, []

    # discard tokens with unsufficient count (or doesn't correspond to tirm_rule)
    for token, count in raw_vocab.items():
      if keep_vocab_item(token, count, min_count, trim_rule=trim_rule):
        retain_tokens.append(token)
        retain_total += count  
      else:          
        drop_unique += 1
        drop_total += count
    
    original_unique_total = len(retain_tokens) + drop_unique
    retain_unique_pct = len(retain_tokens) * 100 / max(original_unique_total, 1)
    logger.info(
      f"min_count={min_count} retains {len(retain_tokens)} unique "
      f"words ({retain_unique_pct}%% of original {original_unique_total}, drops {drop_unique})"
    )

    original_total = retain_total + drop_total
    retain_pct = retain_total * 100 / max(original_total, 1)
    logger.info(
      f"min_count={min_count} leaves {retain_total} token corpus "
      f"({retain_pct}%% of original {original_total}, drops {drop_total})"
    )

    # Precalculate each vocabulary item's threshold for sampling
    if sample <= 1.0:
      # traditional meaning: set parameter as proportion of total
      threshold_count = sample * retain_total
    # else:
    #   # new shorthand: sample >= 1 means downsample all words with higher count than sample
    #   threshold_count = int(sample * (3 + np.sqrt(5)) / 2)

    downsample_total, downsample_unique = 0, 0
    filtered_vocab = defaultdict()
    for w in retain_tokens:
      count = raw_vocab[w]
      token_probability = (np.sqrt(count / threshold_count) + 1) * (threshold_count / count)
      filtered_vocab[w] = (count, token_probability)
      if token_probability < 1.0:
        downsample_unique += 1
        downsample_total += token_probability * count
      else:
        token_probability = 1.0
        downsample_total += count

    logger.info(f"sample={sample} downsamples {downsample_unique} most-common tokens")

    return filtered_vocab


class FrqRandomTokensTrainSampler(RandomTokensTrainSampler):
  def __init__(
    self, corpus_iter: BaseDatasetIterator, 
    tokenizer: AutoTokenizer,
    args: DataArguments,
    ):
    super().__init__(
      corpus_iter, tokenizer, args
      )
    frq_sub_sampling = FrqSubSamplingVocab(
      corpus_iter, tokenizer, args.max_seq_len, args.min_count, args.sample, args.trim_rule, args.train_path
      )
    self.vocab, self.corpus_count = frq_sub_sampling.get_vocab(), frq_sub_sampling.get_corpus_count()

  def _sample(
    self,
    token_ids: List[int], 
    )-> Dict[str, List[int]]:

    tokens_pos = np.array(list(range(1,len(token_ids)+1))) # exclude pos 0 and max_seq_len (CLS and SEP)
    
    # subsampling based on frq 
    dist = np.array([self.vocab.get(tok_id,(0,0.))[1] for tok_id in token_ids])
    mask = self.rng.random() < dist
    retained_token_pos = tokens_pos[mask]

    n_retained_tokens = len(retained_token_pos)

    if n_retained_tokens < max(1, self.min_seq_len): # not enough tokens for one sample
      return {'token_ind': []}
    
    # Sample N random tokens
    # pvals = dist[mask]
    # pvals /= np.sum(pvals) # need to sum to 1
    # # sample the random tokens using the subsampling dist => can use a unifrom one 
    # token_pos = np.sort(self.rng.choice(retained_token_pos, self.N, p=pvals)) # use dist even after subsampling
    # token_pos = np.sort(self.rng.choice(retained_token_pos, self.N)) #uniform over retained tokens (after subsampling)


    token_pos = self.rng.choice(retained_token_pos, min(n_retained_tokens, self.N), replace=False) # no repetition
    m = self.N - len(token_pos)
    if m > 0:
      token_pos = np.concatenate([token_pos, self.rng.choice(retained_token_pos, m)])
    
    return {'token_ind': token_pos.tolist()}

class UniformRandomTokensTrainSampler(RandomTokensTrainSampler):
  def __init__(
    self, corpus_iter: BaseDatasetIterator, 
    tokenizer: AutoTokenizer,
    args: DataArguments,
    ):
    super().__init__(
      corpus_iter, tokenizer, args
      )
    self.corpus_count = sum(1 for _ in self.corpus_iter)

  def _sample(
    self,
    token_ids: List[int], 
    )-> Dict[str, List[int]]:

    tokens_pos = np.array(list(range(1,len(token_ids)+1))) # exclude pos 0 and max_seq_len (CLS and SEP)

    n_tokens = len(tokens_pos)

    if n_tokens < max(1, self.min_seq_len): # not enough tokens for one sample
      return {'token_ind': []}
    
    # Sample N random tokens
    # # sample the random tokens using a unifrom dist 
    # token_pos = np.sort(self.rng.choice(tokens_pos, self.N))

    token_pos = self.rng.choice(tokens_pos, min(n_tokens, self.N), replace=False) # no repetition
    m = self.N - len(token_pos)
    if m > 0:
      token_pos = np.concatenate([token_pos, self.rng.choice(tokens_pos, m)])
    
    return {'token_ind': token_pos.tolist()}

RANDOM_SAMPLING_STRATEGIES={
  'frq': FrqRandomTokensTrainSampler,
  'uni': UniformRandomTokensTrainSampler,
  'none': SequenceTokensTrainSampler,
}

def get_data_sampler(strategy):
  if strategy not in RANDOM_SAMPLING_STRATEGIES:
    raise ValueError(
        f"{strategy} is not supported, only {', '.join(RANDOM_SAMPLING_STRATEGIES.keys())} are supported."
      )
  return RANDOM_SAMPLING_STRATEGIES[strategy]


# Ranking Triplets sampling
# Query, doc+, doc- iterator 
class CollectionReader:
  def __init__(self, data_folder) -> None:
    super().__init__()
    self.corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
    self.collection_filepath = os.path.join(data_folder, 'collection.tsv')
      
    if not os.path.exists(self.collection_filepath):
      tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
      if not os.path.exists(tar_filepath):
        logger.info("Download collection.tar.gz")
        http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

      with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)
          
  def get(self):
    if not self.corpus:
      logger.info("Reading corpus: collection.tsv")
      with open(self.collection_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
          pid, passage = line.strip().split("\t")
          pid = int(pid)
          self.corpus[pid] = passage
    return self.corpus

class QueriesReader:
  def __init__(self, data_folder) -> None:
    super().__init__()
    self.queries = {}        #dict in the format: query_id -> query. Stores all training queries
    self.queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
    if not os.path.exists(self.queries_filepath):
      tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
      if not os.path.exists(tar_filepath):
        logger.info("Download queries.tar.gz")
        http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

      with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

  def get(self):
    if not self.queries:
      logger.info("Reading queries: queries.train.tsv")
      with open(self.queries_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
          qid, query = line.strip().split("\t")
          qid = int(qid)
          self.queries[qid] = query
    return self.queries
  
class TeacherScoresReader:
  """Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
    to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L-6-v2 model"""
  def __init__(self, data_folder) -> None:
    super().__init__()
    self.ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
    if not os.path.exists(self.ce_scores_file):
      logger.info("Download cross-encoder scores file")
      http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz', self.ce_scores_file)
    self.ce_scores = {}

  def get(self, keys=None):
    if not self.ce_scores:
      with gzip.open(self.ce_scores_file, 'rb') as fIn:
        self.ce_scores = pickle.load(fIn)
    if keys is not None:
      return {k: self.ce_scores[k] for k in keys}
    return self.ce_scores

class HardNegativesReader:
  def __init__(self, data_folder) -> None:
    super().__init__()
    # As training data we use hard-negatives that have been mined using various systems
    self.hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives.jsonl.gz')
    if not os.path.exists(self.hard_negatives_filepath):
      logger.info("Download cross-encoder scores file")
      http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz', self.hard_negatives_filepath)

  def get(
    self,
    queries: Dict,
    p_negs_to_use: Optional[list] = None,
    num_negs_per_system: Optional[int] = 5,
    max_passages: Optional[int] = 0,
    use_all_queries: Optional[bool] = False,
    ):
    train_queries = {}
    negs_to_use = None
    with gzip.open(self.hard_negatives_filepath, 'rt') as fIn:
      for line in tqdm.tqdm(fIn):
        if max_passages > 0 and len(train_queries) >= max_passages:
          break
        data = json.loads(line)
        #Get the positive passage ids
        pos_pids = data['pos']

        #Get the hard negatives
        neg_pids = set()
        if negs_to_use is None:
          if p_negs_to_use is not None:    #Use specific system for negatives
            negs_to_use = p_negs_to_use.split(",")
          else:   #Use all systems
            negs_to_use = list(data['neg'].keys())
            logger.info(f"Using negatives from the following systems: {negs_to_use}")

        for system_name in negs_to_use:
          if system_name not in data['neg']:
            continue

          system_negs = data['neg'][system_name]
          negs_added = 0
          for pid in system_negs:
            if pid not in neg_pids:
              neg_pids.add(pid)
              negs_added += 1
              if negs_added >= num_negs_per_system:
                break

        if use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
          train_queries[data['qid']] = {'qid': data['qid'], 'query': queries[data['qid']], 'pos': pos_pids, 'neg': neg_pids}

    logger.info("Train queries: {}".format(len(train_queries)))
    return train_queries

class TripletsDataset(torch.utils.data.Dataset):
    def __init__(
      self, 
      queries, corpus, ce_scores,
      tokenizer,
      args,
      ):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.ce_scores = ce_scores

        self.tokenizer = tokenizer
        self.args = args
        self.seed = args.rng_seed
        self.rng = np.random.default_rng(self.seed)

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            self.rng.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']
        qid = query['qid']

        if len(query['pos']) > 0:
            pos_id = query['pos'].pop(0)    #Pop positive and add at end
            pos_text = self.corpus[pos_id]
            query['pos'].append(pos_id)
        else:   #We only have negatives, use two negs
            pos_id = query['neg'].pop(0)    #Pop negative and add at end
            pos_text = self.corpus[pos_id]
            query['neg'].append(pos_id)

        #Get a negative passage
        neg_id = query['neg'].pop(0)    #Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['neg'].append(neg_id)

        pos_score = self.ce_scores[qid][pos_id]
        neg_score = self.ce_scores[qid][neg_id]

        # Get token_ids and mask_ids
        query_input = self.tokenizer.encode_plus(
          query_text,
          max_length = self.args.max_query_len,
          truncation= True,
          return_attention_mask = True,
          return_token_type_ids = True,
        )
        
        pos_input = self.tokenizer.encode_plus(
          pos_text,
          max_length = self.args.max_seq_len,
          truncation= True,
          return_attention_mask = True,
          return_token_type_ids = True,
        )

        neg_input = self.tokenizer.encode_plus(
          neg_text,
          max_length = self.args.max_seq_len,
          truncation= True,
          return_attention_mask = True,
          return_token_type_ids = True,
        )

        return self.InputSample(query_input, pos_input, neg_input, label=pos_score-neg_score).to_features()

    def __len__(self):
        return len(self.queries)
    
    class InputSample(object):
      """ A sinlge Train instance """
      def __init__(self, 
        query_input: BatchEncoding,
        pos_input: BatchEncoding,
        neg_input: BatchEncoding,
        label: float,
        ) -> None:

        self.query_input = query_input
        self.pos_input = pos_input
        self.neg_input = neg_input
        self.label = label
      
      def to_features(self):
        features = {
          "query_inputs": self.query_input,
          "pos_inputs": self.pos_input, 
          "neg_inputs": self.neg_input,
          "labels": self.label,
        }
        return features

@dataclass
class TripletsCollator:
    """
    Data collator that will dynamically pad the inputs received.
    """
    tokenizer: AutoTokenizer
    max_query_len: int = 50
    max_seq_len: int = 300

    def __call__(
            self, features
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:

      query_inputs = [f['query_inputs'] for f in features]
      query_inputs = self.tokenizer.pad(
          query_inputs,
          padding='longest',
          max_length=self.max_query_len,
          return_tensors='pt'
      )

      pos_inputs = [f['pos_inputs'] for f in features]
      pos_inputs = self.tokenizer.pad(
          pos_inputs,
          padding='longest',
          max_length=self.max_seq_len,
          return_tensors='pt'
      )

      neg_inputs = [f['neg_inputs'] for f in features]
      neg_inputs = self.tokenizer.pad(
          neg_inputs,
          padding='longest',
          max_length=self.max_seq_len,
          return_tensors='pt'
      )

      labels = torch.Tensor([f['labels'] for f in features])
      
      return {
          "query_inputs": query_inputs,
          "pos_inputs": pos_inputs, 
          "neg_inputs": neg_inputs,
          "labels": labels,
        }
