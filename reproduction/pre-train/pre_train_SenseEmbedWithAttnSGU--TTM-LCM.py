# coding=utf-8
import logging
import os
import sys

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from src.arguments import PosModelArguments as ModelArguments, DataArguments, TrainingArguments
from src.models import SenseEmbedwithAttnSGUFromTransformer as TTM_LCM
from src.losses import EmbLossTransformerEmb as Loss
from src.trainers import Trainer
from src.trainer_utils import HTensorBoardCallback, LossPrinterCallback
from src.data import (
    MsMarcoIterator, 
    RandomTokenCollator, 
    get_data_sampler,
)

logger = logging.getLogger(__name__)


def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
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

    num_labels = 1

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels = num_labels,
        cache_dir = model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir = model_args.cache_dir,
        use_fast = model_args.use_fast_tokenizer,
    )
    # Load or initialize a SeneEmbed model
    inner_model = TTM_LCM.from_pretrained(
        model_args.model_name_or_path, # This path contains either: Transformers name_or_path to hub model, or local checkpoint of SenseEmbed (with the complete model)
        from_tf = bool(".ckpt" in model_args.model_name_or_path),
        config = config,
        cache_dir = model_args.cache_dir,
        model_args = model_args,
        freeze_transformer_weights = training_args.freeze_transformer_weights,
    )
    # Loss model
    model = Loss(
        model = inner_model,
        args = model_args,
    )

    # Get datasets
    callbacks = []
    if training_args.do_train:
        callbacks.append(LossPrinterCallback())
        text_iter = MsMarcoIterator(data_args.train_path)
        data_sampler = get_data_sampler(data_args.sampling_strategy)
        train_dataset = data_sampler(
            text_iter, tokenizer, data_args
        )
        if training_args.log_gradients:
            callbacks.append(HTensorBoardCallback())
    else:
        train_dataset = None

    # Initialize our Trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        data_collator = RandomTokenCollator(
            tokenizer,
            data_args.max_seq_len,
        ),
        callbacks = callbacks,
    )

    # Training
    if training_args.do_train:
        resume = training_args.resume_from_checkpoint if training_args.resume_from_checkpoint else True
        trainer.train(
            resume_from_checkpoint = resume, # model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None,  # can use the model_args.model_name_or_path directly| True will resume training from the last checkpoint in output_dir
        )
        # trainer.save_model()
        # # For convenience, we also re-save the tokenizer to the same directory,
        # # so that you can share your model easily on huggingface.co/models =)
        # if trainer.is_world_process_zero():
        #     tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    # online wandb change here
    os.environ["WANDB_MODE"]='offline'
    main()