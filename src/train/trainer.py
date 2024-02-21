import logging
import os
from itertools import repeat
from typing import Any, Dict, List, Optional, Tuple, Union
from other_arguments import otherArguments
import random
from transformers import BatchEncoding, PreTrainedTokenizer
from openmatch.trainer import DRTrainer
from openmatch.dataset.train_dataset import TrainDatasetBase, MappingTrainDatasetMixin,StreamTrainDatasetMixin
import torch.nn as nn
import torch
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import Trainer
from torch.utils.data import Dataset
from openmatch.arguments import DataArguments
import glob

from tqdm import tqdm
logger = logging.getLogger(__name__)
import jsonlines

class STrainer(DRTrainer):
    def __init__(self, other_args,*args, **kwargs):
        super(STrainer, self).__init__(*args, **kwargs)
        self.other_args = other_args

    def compute_loss(self, model, inputs, return_outputs=False):

        compression_input,label,data_type= inputs
        outputs = model(compression_input=compression_input,label=label,task_type=data_type)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs,
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        with torch.no_grad():
            with self.autocast_smart_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

        return (loss, None, None)



    def create_optimizer(self):

        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model.lm_q, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.other_args.fixed_decoder:
                fixed_parameters_name = ["decoder"]
                fixed_parameters = [name for name, parameters in opt_model.lm_q.named_parameters()
                                    if any(char in name for char in fixed_parameters_name)]

                for name, param in opt_model.lm_q.named_parameters():
                    if name in fixed_parameters:
                        param.requires_grad = False

                for name, param in opt_model.teacher_model.named_parameters():
                        param.requires_grad = False



                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.lm_q.named_parameters() if n not in fixed_parameters],
                        "weight_decay": self.args.weight_decay,
                    }, ]

                model_parameters = [n for n, p in opt_model.lm_q.named_parameters()]
                optimizer_name = [{
                    "params": [n for n, p in opt_model.lm_q.named_parameters() if n in decay_parameters],
                }, {
                    "params": [n for n, p in opt_model.lm_q.named_parameters() if n not in decay_parameters]
                }, ]

                print("-----------------optimizer parameters without decoder----------------------------")

            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.lm_q.named_parameters()],
                        "weight_decay": self.args.weight_decay,
                    }, ]

                model_parameters = [n for n, p in opt_model.lm_q.named_parameters()]
                optimizer_name = [{
                    "params": [n for n, p in opt_model.lm_q.named_parameters() if n in decay_parameters],
                }, {
                    "params": [n for n, p in opt_model.lm_q.named_parameters() if n not in decay_parameters]
                }, ]

                print("-----------------optimizer all parameters----------------------------")



            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

class NewDataset(Dataset):

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            data_args: DataArguments,
            other_args: otherArguments,
            trainer: DRTrainer = None,
            is_eval: bool = False,

    ) -> None:
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.trainer = trainer
        self.is_eval = is_eval
        self.other_args = other_args
        self.data = self.load_data(data_args, is_eval)

    def load_data(self, data_args, is_eval):

        data = []
        if is_eval ==False:
            with open(data_args.train_path, 'r') as f:
                reader = jsonlines.Reader(f)
                for line in tqdm(reader):
                    data.append(line)
        else:
            with open(data_args.eval_path, 'r') as f:
                reader = jsonlines.Reader(f)
                for line in tqdm(reader):
                    data.append(line)
        return data

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        example = self.data[index]
        query = example['instruction'] + example['input']
        label = example['output']

        data_type = example['compression_type']

        if example['compression_type']=='passage':
            compression_input = example['passage'][0]+' '+query
        elif example['compression_type']=='instruction':
            compression_input = query

        return {'compression_input': compression_input, 'query':query, 'label':label,'data_type':data_type}

