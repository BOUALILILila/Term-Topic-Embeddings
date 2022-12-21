import os
from transformers.integrations import TensorBoardCallback, rewrite_logs
from transformers.trainer_callback import (
    TrainerCallback
)
from transformers.utils import logging


logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

class HTensorBoardCallback(TensorBoardCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            if 'model' in kwargs:
                model = kwargs.get('model')
                for tag, value in model.named_parameters():
                    if tag.startswith('transformer_model.'): 
                      continue
                    tag = tag.replace(".", "/")
                    self.tb_writer.add_histogram(tag, value.data.cpu().numpy(), state.global_step)
                    if value.grad is not None:
                        self.tb_writer.add_histogram(tag + "/grad", value.grad.data.cpu().numpy(), state.global_step)
                    # else:
                    #     logger.warning(f">>> model parameter: {tag}, {type(value)}, {type(value.grad)}")

            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()

class LossPrinterCallback(TrainerCallback):
    """
    A bare :class:`~transformers.TrainerCallback` that just prints the loss to a tsv file.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        self.writer = open(os.path.join(args.output_dir,'train_loss.tsv'), 'a')
        self.buff = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if self.writer is not None:
            loss = logs['loss'] if 'loss' in logs else None
            if loss is not None:
                self.buff.append(f"{state.global_step}\t{loss}\n")
    
    def on_save(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if self.buff and self.writer is not None:
            self.writer.writelines(self.buff)
            self.writer.flush()
            self.buff = []

    def on_train_end(self, args, state, control, **kwargs):
        if self.writer:
            if self.buff:
                self.writer.writelines(self.buff)
            self.writer.close()
            self.writer = None