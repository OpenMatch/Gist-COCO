import copy
import importlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (AutoConfig, AutoModel, BatchEncoding,
                          PreTrainedModel, T5EncoderModel, T5ForConditionalGeneration,AutoModelForSeq2SeqLM)
from transformers.modeling_outputs import ModelOutput
from transformers import AutoTokenizer
from openmatch.arguments import DataArguments
from openmatch.arguments import DRTrainingArguments as TrainingArguments
from openmatch.arguments import ModelArguments
from other_arguments import otherArguments
from openmatch.utils import mean_pooling
from openmatch.modeling.linear import LinearHead
from label_prompt_model import Label_prompt_T5ForConditionalGeneration
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DROutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None


class SModel(nn.Module):
    def __init__(
            self,
            lm_q: PreTrainedModel,
            teacher_model: PreTrainedModel,
            tied: bool = True,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
            other_args: otherArguments = None,
            tokenizer=None,
    ):
        super().__init__()

        self.tied = tied
        self.lm_q = lm_q
        self.teacher_model=teacher_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.other_args = other_args
        self.tokenizer = tokenizer

        if train_args is not None:
            self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
            if train_args.negatives_x_device:
                if not dist.is_initialized():
                    raise ValueError('Distributed training has not been initialized for representation all gather.')
                self.process_rank = dist.get_rank()
                self.world_size = dist.get_world_size()

    # def _get_config_dict(self):
    #     config = {
    #         "tied": self.tied,
    #         "plm_backbone": {
    #             "type": type(self.lm_q).__name__,
    #             "feature": self.feature,
    #         },
    #         "pooling": self.pooling,
    #         "linear_head": bool(self.head_q),
    #         "normalize": self.normalize,
    #     }
    #     return config

    def forward(
            self,
            compression_input = None,
            label = None,
            task_type =None
    ):

        self.teacher_model.eval()
        loss = self.label_kl_parament(compression_input, label, task_type)


        if self.training and self.train_args.negatives_x_device:
            loss = loss * self.world_size  # counter average weight reduction
        return DROutput(
            loss=loss,
        )



    def generate_zeros_with_ones_like(self,input, n):
        dimensions = input.size()
        zeros_tensor = torch.zeros(dimensions).to(input.device)
        add_attention_mask = torch.ones(input.size(0), n).to(input.device)
        cross_attention_mask = torch.cat((add_attention_mask, zeros_tensor), dim=1)
        return cross_attention_mask.to(input.device)


    def KL_loss_parament(self,augmented_query_inputs, short_label,model, teacher_model,task_type):

        instruction_input_list=[]
        instrcution_label_list=[]

        passage_input_list =[]
        passage_label_list =[]

        passage_kl_loss=0
        instruction_kl_loss=0
        for idx, type in enumerate(task_type):
            if type=='instruction':
                instruction_input_list.append(augmented_query_inputs[idx])
                instrcution_label_list.append(short_label[idx])
            else:
                passage_input_list.append(augmented_query_inputs[idx])
                passage_label_list.append(short_label[idx])

        if instruction_input_list != []:
            instruction_input = self.tokenizer(instruction_input_list, padding='max_length', max_length=self.data_args.q_max_len,
                                           return_tensors="pt", truncation=True).to(model.device)
            instrcution_label = self.tokenizer(instrcution_label_list, padding='max_length', max_length=self.other_args.sl_max_len,
                                           return_tensors="pt", truncation=True, add_special_tokens=False).to(model.device)

            instrcution_label_collate = instrcution_label.input_ids
            instrcution_label_collate[instrcution_label_collate == self.tokenizer.pad_token_id] = -100

            instrcution_cross_attention_mask = self.generate_zeros_with_ones_like(instruction_input.attention_mask,
                                                                              self.other_args.num_prompt)

            instruction_input = BatchEncoding(instruction_input)
            student_items_out = model(**instruction_input, return_dict=True, labels = instrcution_label_collate,
                                      cross_attention_mask=instrcution_cross_attention_mask,
                                      use_task_prompt=True,)

            with torch.no_grad():
                teacher_items_out = teacher_model(**instruction_input, return_dict=True, labels = instrcution_label_collate)

            counts = []
            for i in range(instrcution_label_collate.size(0)):
                count = torch.sum(instrcution_label_collate[i, :] != -100)
                counts.append(count.item())

            instruction_kl_loss = self.get_kl(student_items_out.logits, teacher_items_out.logits, task_type, counts)

        if passage_input_list!=[]:
            passage_input =self.tokenizer(passage_input_list ,padding='max_length',max_length=self.data_args.p_max_len,
                                          return_tensors="pt",truncation=True).to(model.device)
            passage_label = self.tokenizer(passage_label_list,padding='max_length',max_length = self.other_args.sl_max_len,
                                           return_tensors="pt",truncation=True,add_special_tokens=False).to(model.device)

            passage_label_collate = passage_label.input_ids
            passage_label_collate[passage_label_collate == self.tokenizer.pad_token_id] = -100

            passage_cross_attention_mask = self.generate_zeros_with_ones_like(passage_input.attention_mask, self.other_args.num_prompt*2)
            passage_input=BatchEncoding(passage_input)
            student_items_out = model(**passage_input, return_dict=True, labels=passage_label_collate,
                                      cross_attention_mask=passage_cross_attention_mask,
                                      use_knowledge_prompt=True,)

            with torch.no_grad():
                teacher_items_out = teacher_model(**passage_input, return_dict=True, labels=passage_label_collate)

            counts = []
            for i in range(passage_label_collate.size(0)):
                count = torch.sum(passage_label_collate[i, :] != -100)
                counts.append(count.item())

            passage_kl_loss = self.get_kl(student_items_out.logits, teacher_items_out.logits, task_type, counts)



        all_loss = instruction_kl_loss+passage_kl_loss
        return all_loss.requires_grad_(True)


    def label_kl_parament(self, compression_input, label,task_type):
        return self.KL_loss_parament(compression_input, label,
                                     self.lm_q, self.teacher_model,task_type)

    def get_kl(self, student_logits, teacher_logits,task_type,counts):
        idx=[]
        pad_length=[]
        for id in range(len(counts)):
                idx.append(id)
                pad_length.append(counts[id])
        indices = torch.tensor(idx)



        nq_student_logits = student_logits[indices,:,:]
        nq_teacher_logits = teacher_logits[indices, :, :]

        nq_student_logits=nq_student_logits.view(nq_student_logits.size(0)*nq_student_logits.size(1),-1)
        nq_teacher_logits=nq_teacher_logits.view(nq_teacher_logits.size(0)*nq_teacher_logits.size(1),-1)


        nq_student_logits_probs = torch.nn.functional.log_softmax(nq_student_logits, dim=-1)
        nq_teacher_logits_probs = torch.nn.functional.softmax(nq_teacher_logits, dim=-1)

        kl = torch.nn.KLDivLoss(reduction="none")
        nq_kl_loss_all = kl(nq_student_logits_probs, nq_teacher_logits_probs)
        mean_nq_kl_loss = torch.sum(nq_kl_loss_all, dim=-1)

        pad_mask_tensor = torch.ones(len(pad_length), student_logits.size(1))

        for ii, num in enumerate(pad_length):
            pad_mask_tensor[ii, num:] = 0
        pad_mask_tensor = pad_mask_tensor.view(pad_mask_tensor.size(0) * pad_mask_tensor.size(1))

        mask = torch.eq(pad_mask_tensor, 1)
        mask=mask.to(nq_student_logits.device)
        selected_loss = torch.masked_select(mean_nq_kl_loss, mask)


        nq_kl_loss = torch.mean(selected_loss, dim=-1)

        #kl1=torch.nn.KLDivLoss(reduction="batchmean")
        #kl_l = kl1(nq_student_logits_probs, nq_teacher_logits_probs)


        nq_kl_loss=nq_kl_loss.requires_grad_(True)
        nq_kl_loss=nq_kl_loss.to(nq_student_logits.device)

        return nq_kl_loss


    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            model_name_or_path: str = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
            other_args: otherArguments = None,
            **hf_kwargs,
    ):
        model_name_or_path = model_name_or_path or model_args.model_name_or_path
        teacher_model_name_or_path=other_args.teacher_model_name_or_path
        auxiliary_tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,)


        tied = not model_args.untie_encoder
        model_class = Label_prompt_T5ForConditionalGeneration
        lm_q = model_class.from_pretrained(model_name_or_path, **hf_kwargs)

        prompt_token = auxiliary_tokenizer.additional_special_tokens_ids[::-1]
        lm_q.set_up_task_prompt(other_args.num_prompt,prompt_token)

        prompt_token = auxiliary_tokenizer.additional_special_tokens_ids
        lm_q.set_up_knowledge_prompt(other_args.num_prompt, prompt_token)
        print("------------------------initial_task_compression_model-------------------------------")

        if other_args.train_LM_out_label:
            teacher_model=None
            print("------------------------train_LM_out_label-------------------------------")
        else:
            teacher_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_model_name_or_path)
            print("------------------------train_KL-------------------------------")




        model = cls(
            lm_q=lm_q,
            tied=tied,
            teacher_model=teacher_model,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
            other_args=other_args,
            tokenizer=auxiliary_tokenizer
        )
        return model

    def save(self, output_dir: str):

        self.lm_q.save_pretrained(output_dir)

        if self.lm_q.task_prompt is not None:
            np.save(os.path.join(output_dir, 'task_prompt.npy'), self.lm_q.task_prompt.clone().detach().cpu().numpy())
            print("----------------------------------save task_prompt successful-----------------------------")

        if self.lm_q.knowledge_prompt is not None:
            np.save(os.path.join(output_dir, 'knowledge_prompt.npy'),self.lm_q.knowledge_prompt.clone().detach().cpu().numpy())
            print("----------------------------------save knowledge_prompt successful-----------------------------")

        # with open(os.path.join(output_dir, 'openmatch_config.json'), 'w') as f:
        #     json.dump(self._get_config_dict(), f, indent=4)


