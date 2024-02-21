import copy
import warnings
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
from transformers.models.t5 import T5ForConditionalGeneration
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    __HEAD_MASK_WARNING_MSG,
    T5Attention,
    T5Block,
    T5LayerCrossAttention,
    T5LayerFF,
    T5LayerNorm,
    T5LayerSelfAttention,
    T5PreTrainedModel,
    T5Stack,
    checkpoint,
)
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
import numpy as np
logger = logging.get_logger(__name__)


class Label_prompt_T5ForConditionalGeneration(T5ForConditionalGeneration):



    def __init__(self, config: T5Config):
        """Same as existing init, but uses GistT5Stack"""
        super(T5PreTrainedModel, self).__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.task_prompt = None
        self.knowledge_prompt = None

        self.adaptor_layer_first = nn.Linear(config.d_model, config.d_model)



    def set_up_task_prompt(self, k, prompt_id_list):
        self.prompt_tuning_k = k
        # Initialize with special embeddings
        k_idxs = prompt_id_list[:k]
        self.task_prompt = nn.Parameter(
            self.shared.weight[k_idxs].clone().detach().unsqueeze(0)
        )
        print("initial task prompt embedding")

    def set_up_knowledge_prompt(self, k, prompt_id_list):
        self.prompt_tuning_k = k
        # Initialize with special embeddings
        k_idxs = prompt_id_list[:k]
        self.knowledge_prompt = nn.Parameter(
            self.shared.weight[k_idxs].clone().detach().unsqueeze(0)
        )
        print("initial knowledge prompt embedding")



    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cross_attention_mask: Optional[torch.BoolTensor] = None,

        use_task_prompt: Optional[bool] = False,
        use_knowledge_prompt: Optional[bool] = False,
        gist_idx: Optional = None,
        no_cross_attention_mask: Optional[bool] = True,
        use_adaptor: Optional[bool] = False,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
            Indices should be in `[-100, 0, ..., config.vocab_size - 1]`. All
            labels set to `-100` are ignored (masked), the loss is only computed
            for labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park",
                return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the
                <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for
                you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args -
        # head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask



        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed


            if use_task_prompt:
                self.task_prompt = self.task_prompt.to(input_ids.device)

                raw_inputs_embeds = self.encoder.embed_tokens(input_ids)
                new_inputs_embeds = torch.cat((self.task_prompt.expand(raw_inputs_embeds.size(0), -1, -1),
                                               raw_inputs_embeds), dim=1)

                add_attention_mask = torch.ones(attention_mask.size(0), self.prompt_tuning_k).to(attention_mask.device)
                new_attention_mask = torch.cat((add_attention_mask, attention_mask), dim=1)
                attention_mask = new_attention_mask
                input_ids = None


            elif use_knowledge_prompt:
                self.knowledge_prompt = self.knowledge_prompt.to(input_ids.device)
                self.task_prompt = self.task_prompt.to(input_ids.device)

                raw_inputs_embeds = self.encoder.embed_tokens(input_ids)
                new_inputs_embeds = torch.cat(
                    (self.knowledge_prompt.expand(raw_inputs_embeds.size(0), -1, -1), self.task_prompt.expand(raw_inputs_embeds.size(0), -1, -1),
                     raw_inputs_embeds),dim=1)

                add_attention_mask = torch.ones(attention_mask.size(0), self.prompt_tuning_k*2).to(attention_mask.device)
                new_attention_mask = torch.cat((add_attention_mask, attention_mask), dim=1)
                attention_mask = new_attention_mask
                input_ids = None

            else:
                new_inputs_embeds=None


            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=new_inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]


        if use_knowledge_prompt:
            hidden_states = hidden_states[:, :self.prompt_tuning_k*2, :]
            cross_attention_mask = cross_attention_mask[:, :self.prompt_tuning_k*2]

        if use_task_prompt:
            hidden_states = hidden_states[:, :self.prompt_tuning_k, :]
            cross_attention_mask = cross_attention_mask[:, :self.prompt_tuning_k]




        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        if cross_attention_mask is None:
            cross_attention_mask = attention_mask


        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if cross_attention_mask is None:
                cross_attention_mask = attention_mask
            cross_attention_mask = cross_attention_mask.to(self.decoder.first_device)

            # If cross attention mask is supplied, then set the right mask.
            # Otherwise reuse the attention mask.

        # Decode
        # ==== BEGIN GIST CHANGES ====
        # `attention_mask` right now is (batch_size, input_seq_len,
        # input_seq_len) - it should be (batch_size, target_seq_len,
        # input_seq_len), where every row of the cross attention mask is the
        # gisted version (i.e. cannot attend to prior tokens).
        # The problem is that it's somewhat difficult to extract the specific
        # row(s) of `attention_mask` corresponding to the gisted condition since
        # it's not clear what is padding and what is the gist-modified mask. So
        # our model accepts an additional argument, `cross_attention_mask`,
        # which should be (batch_size, target_seq_len, input_seq_len), which
        # dictates how target seq tokens can attend back to input tokens.
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=cross_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # ==== END GIST CHANGES ====

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586  # noqa
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666  # noqa

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        outputs = Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        return outputs


