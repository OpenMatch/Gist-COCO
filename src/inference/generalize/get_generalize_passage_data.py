import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from ..model.label_prompt_model import Label_prompt_T5ForConditionalGeneration
import numpy as np
import os
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
import torch.nn.functional as F

import random
completion_template = "Q: {} A:"
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
qa_templete = "In this task, you're given a question. Your task is to provide a short answer to the given question. Question:{} Answer:"

import datetime
import json
import math
from concurrent.futures.process import ProcessPoolExecutor

import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizerBase
from torch.utils.data import DataLoader



class MultiEncoderDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        chunk_size: int,
        num_samples: int = None,
        verbose: bool = False,
        ignore_pad_token_for_loss: bool = True,
        max_workers=1,
        num_passage =1,
    ):

        self.tokenizer = tokenizer
        self.chunk_size=chunk_size
        self.num_passage  =num_passage
        self.num_samples = num_samples
        self.verbose = verbose
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self._encode_data(data_path, max_workers)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, index):
        return self.encodings[index]

    def my_collate(self,batch):
        # input_ids_list = [item['input_ids'] for item in batch]
        # attention_mask_list = [item['attention_mask'] for item in batch]

        labels = [item['labels'] for item in batch]
        raw_query = [item['raw_query'] for item in batch]
        passages = [item['passages'] for item in batch]
        now_passage = [item['now_passage'] for item in batch]
        now_passage = sum(now_passage, [])

        output = self.tokenizer(now_passage, return_tensors="pt", max_length=self.chunk_size,
                                padding="longest", truncation=True)

        input_ids = output['input_ids']
        attention_mask = output['attention_mask']
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'raw_query': raw_query,
                'passages': passages}

    def _encode_data(self, file_path, max_workers):
        with open(file_path) as f:
            if max_workers == 1:
                encodings = list(map(self._process_line, enumerate(f)))
            else:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    encodings = executor.map(self._process_line, enumerate(f))
        self.encodings = [enc for enc in encodings if enc is not False]
        if self.num_samples is not None:
            assert self.num_samples == len(self.encodings)

    def _process_line(self, index_line):
        i, line = index_line
        if i % 100 == 0:
            print('Processed', i, 'records', datetime.datetime.now())
        if self.num_samples is not None and i >= self.num_samples:
            return False
        data = json.loads(line)

        query = data['question']
        if 'output' in data:
            target = data['output']
        else:
            target = data['answers']
        source = data['passages'][:self.num_passage]
        passages = data['passages']


        encoding = self._encode_example(
            source,
            target,
            passages,
            query
        )
        if self.verbose and i == 0:
            print('First record in dataset:')
            for token_ids in encoding['input_ids']:
                print()
                print(self.tokenizer.decode(token_ids))
        return encoding

    def _encode_example(self, source, target, passages=None, query=None):

        now_passage = [p+ ' '+ query for p in source]

        return {
            'labels': target,
            'passages': passages,
            'raw_query': query,
            'now_passage':now_passage
        }



def call_model(prompt, model, tokenizer, device, auxiliary_model,max_new_tokens=15):
    inpts = tokenizer(prompt, return_tensors="pt").to(device)
    gen = model.generate(input_ids=inpts.input_ids, attention_mask=inpts.attention_mask, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(gen[0],skip_special_tokens=True)
    pred = text
    return pred,text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='/data4/flan-t5/flan-t5-large')
    parser.add_argument('--input_file', type=str,default='/data1/lixinze/gist/ours/test_kilt/new-retriever/ra_ance_kilt_wiki/nq_dev_ance_wiki_top10.jsonl')
    parser.add_argument('--auxiliary_model', type=str, default="/data3/lixinze/compression_model/task_knowledge_prompt/ACL_compression/main_two/flant5_large_new_instruction_data_10/flan-t5/checkpoint-44000")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--output_path', type=str, default="/data3/lixinze/compression_model/task_knowledge_prompt/ACL_compression/open_code_check/nq_5p.jsonl")
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--prompt_k', type=int, default=10)
    parser.add_argument('--num_passage', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()
    device = args.device

    prompt_k = args.prompt_k
    auxiliary_model = Label_prompt_T5ForConditionalGeneration.from_pretrained(args.auxiliary_model).eval().to(device)
    auxiliary_tokenizer = AutoTokenizer.from_pretrained(args.auxiliary_model)

    foo = torch.Tensor(np.load(os.path.join(args.auxiliary_model, 'task_prompt.npy')))
    auxiliary_model.task_prompt = foo

    soo = torch.Tensor(np.load(os.path.join(args.auxiliary_model, 'knowledge_prompt.npy')))
    auxiliary_model.knowledge_prompt = soo

    auxiliary_model.prompt_tuning_k = prompt_k
    auxiliary_model.eval()
    print('model is load...')

    input_path = args.input_file
    print('data is load...')

    chunk_size = 256


    d = MultiEncoderDataset(
        data_path=input_path,
        tokenizer=auxiliary_tokenizer,
        chunk_size=chunk_size,
        num_passage=args.num_passage
    )


    device = "cuda"
    my_collate = d.my_collate
    dataloader = DataLoader(d, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate)

    prompt_k = prompt_k * 2
    final_attentions =[[] for _ in range(5)]
    with open(args.output_path, 'w') as f:
        for batch in tqdm(dataloader):
            labels = batch.pop('labels')
            raw_query = batch.pop('raw_query')
            passages = batch.pop('passages')

            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)

            with torch.no_grad():
                # T5 encoder to decoder
                decoder_input_ids = torch.zeros((batch['input_ids'].shape[0], 1), dtype=torch.long)
                items_out = auxiliary_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    decoder_input_ids=decoder_input_ids.cuda(),
                    output_hidden_states=True,
                    return_dict=True,
                    use_knowledge_prompt=True,
                    output_attentions =True
                )
            hidden = items_out.encoder_last_hidden_state
            gist_hidden = hidden[:, :prompt_k, :]

            v = items_out.encoder_attentions
            attentions = torch.zeros_like(v[0])
            for layer in v:
                attentions += layer

            top_attentions =[]
            for top in range(5):
                top_attentions.append(attentions[top::5])

            for idxx,top_atten in enumerate(top_attentions):
                attention_mean = torch.mean(top_atten, dim=1)
                attention_mean = attention_mean[0, :, :]
                lsm = F.softmax(attention_mean, dim=1)
                lsm = lsm.cpu().detach().numpy()
                final_attentions[idxx].append(lsm)

            now_gist_hidden =  gist_hidden.reshape(int(gist_hidden.size()[0]/args.num_passage), prompt_k*args.num_passage,-1)
            with torch.no_grad():
                new_encoder_outputs = BaseModelOutput(last_hidden_state=now_gist_hidden, hidden_states=None,
                                                      attentions=None,)
                output = auxiliary_model.generate(encoder_outputs=new_encoder_outputs, max_new_tokens=15)

            for ids, tokens in enumerate(output):
                group = {}
                out_seq = auxiliary_tokenizer.decode(tokens, skip_special_tokens=True)
                group['prompts'] = raw_query[ids]
                group['passages'] = passages[ids]
                group['compression_passage'] = out_seq
                group['labels'] = labels[ids]
                f.write(json.dumps(group) + '\n')

if __name__ == "__main__":
    main()