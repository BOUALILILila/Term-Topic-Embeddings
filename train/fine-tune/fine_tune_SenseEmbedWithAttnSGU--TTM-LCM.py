# coding=utf-8
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional


import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from arguments import TripletsDataArguments as DataArguments, TrainingArguments
from models import SSenseEmbedwithAttnSGUFromTransformer as TTM_LCM
from losses import MarginMSELoss as Loss
from trainers import Trainer
from trainer_utils import HTensorBoardCallback, LossPrinterCallback
from data import (
    CollectionReader,
    HardNegativesReader,
    QueriesReader,
    TeacherScoresReader,
    TripletsDataset,
    TripletsCollator,
)

logger = logging.getLogger(__name__)

@dataclass
class Arguments:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to model"}
    )
    tokenizer_name: Optional[str] = field(
        default='distilbert-base-uncased', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    query_model : str = field(
        default="same", metadata={"help": "Query model (same, oracle)"}
    )
    similarity_fct: str = field(
        default='dot', metadata={"help": "The similarity function to use among ('dot','cos')"}
    )
    freeze_subembeddings: bool = field(
        default=True, metadata={"help": "Freeze the subembeddings lookup table during ft"}
    )

def main():

    parser = HfArgumentParser((Arguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model params {model_args}")

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Set seed
    set_seed(training_args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
    )

    # Load or initialize a SeneEmbed model
    inner_model = TTM_LCM.from_pretrained(
        model_args.model_name_or_path,
        freeze_transformer_weights = training_args.freeze_transformer_weights,
    )

    if model_args.freeze_subembeddings:
        inner_model.sense_embeds.requires_grad = False
    
    print(">> sense_embeds.requires_grad = ", inner_model.sense_embeds.requires_grad)
    
    # Loss model
    model = Loss(
        model = inner_model,
        query_model = model_args.query_model, #"oracle", # asym
        args = model_args,
    )

    # Get datasets
    callbacks = []
    if training_args.do_train:
        callbacks.append(LossPrinterCallback())

        collection = CollectionReader(data_args.data_folder).get()
        queries = QueriesReader(data_args.data_folder).get()
        ce_scores = TeacherScoresReader(data_args.data_folder).get()
        train_queries = HardNegativesReader(data_args.data_folder).get(
            queries, data_args.negs_to_use, data_args.num_negs_per_system, data_args.max_passages, data_args.use_all_queries
        )
        train_dataset = TripletsDataset(train_queries, collection, ce_scores, tokenizer, data_args)

        if training_args.log_gradients:
            callbacks.append(HTensorBoardCallback())
    else:
        train_dataset = None

    # Initialize our Trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        data_collator = TripletsCollator(
            tokenizer = tokenizer,
            max_query_len = data_args.max_query_len,
            max_seq_len= data_args.max_seq_len,
        ),
        callbacks = callbacks,
    )

    # Training
    if training_args.do_train:
        resume = training_args.resume_from_checkpoint if training_args.resume_from_checkpoint else True
        trainer.train(
            resume_from_checkpoint = resume, 
        )
        
        if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
        ):
            import shutil

            file_names = os.listdir(training_args.output_dir)
            ckpts = [int(file_name.split('-')[-1]) for file_name in file_names if file_name.split('-')[0]=='checkpoint']
            ckpts.sort()
            ckpts_to_remove = ckpts[:-2]
            for ckpt in ckpts_to_remove:
                shutil.rmtree(os.path.join(training_args.output_dir,f'checkpoint-{ckpt}'))

            
        # trainer.save_model()
        # # For convenience, we also re-save the tokenizer to the same directory,
        # # so that you can share your model easily on huggingface.co/models =)
        # if trainer.is_world_process_zero():
        #     tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()