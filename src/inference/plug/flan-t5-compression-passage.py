import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from inference.model.label_prompt_model import Label_prompt_T5ForConditionalGeneration
from inference.model.T5_passage_decoder import deT5ForConditionalGeneration
import numpy as np
import os
import random

import datetime
import json
from concurrent.futures.process import ProcessPoolExecutor
import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizerBase
from torch.utils.data import DataLoader

completion_template = "Q: {} A:"

qa_templete = "In this task, you're given a question. Your task is to provide a short answer to the given question. Question:{} Answer:"
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)






def get_gold_answers(gold):
    ground_truths = set()
    for item in gold:
        if "answer" in item and item["answer"] and len(item["answer"].strip()) > 0:
            ground_truths.add(item["answer"].strip())
    return ground_truths

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
        args = None
    ):

        self.tokenizer = tokenizer
        self.chunk_size=chunk_size
        self.num_passage  =num_passage
        self.num_samples = num_samples
        self.verbose = verbose
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.args=args
        self._encode_data(data_path, max_workers)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, index):
        return self.encodings[index]

    def my_collate(self,batch):
        now_passage_list = [item['now_passage'] for item in batch]

        labels = [item['labels'] for item in batch]
        raw_query = [item['raw_query'] for item in batch]
        passages = [item['passages'] for item in batch]

        now_passage = sum(now_passage_list, [])
        output = self.tokenizer(now_passage, return_tensors="pt", max_length=self.chunk_size,
                                padding="longest", truncation=True)

        input_ids = output['input_ids']
        attention_mask = output['attention_mask']
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'raw_query': raw_query,
                'passages': passages,"now_passage":now_passage}

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

        if 'prompt' in data:
            # POPQA dataset has it's official templete: "Q: {} A:":
            query = data['prompt']
        else:
            query = qa_templete.format(data['question'])

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
        if self.args.compression:
            now_passage = [p + ' ' + query for p in source]
        else:
            if self.args.with_passage:
                now_passage = [' '.join(source) + ' ' + query]

            else:
                now_passage = [query]

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
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--input_file', type=str,default=None)
    parser.add_argument('--auxiliary_model', type=str, default=None)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--num_passage', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--prompt_k', type=int, default=10)
    parser.add_argument('--compression', type=bool, default=True)
    parser.add_argument('--with_passage', type=bool, default=True)

    args = parser.parse_args()
    args_dict = vars(args)
    device = args.device
    print(args_dict)

    prompt_k = args.prompt_k
    auxiliary_model = Label_prompt_T5ForConditionalGeneration.from_pretrained(args.auxiliary_model).eval().to(
        device)
    auxiliary_tokenizer = AutoTokenizer.from_pretrained(args.auxiliary_model)

    foo = torch.Tensor(np.load(os.path.join(args.auxiliary_model, 'task_prompt.npy')))
    auxiliary_model.task_prompt = foo

    soo = torch.Tensor(np.load(os.path.join(args.auxiliary_model, 'knowledge_prompt.npy')))
    auxiliary_model.knowledge_prompt = soo

    auxiliary_model.prompt_tuning_k = prompt_k
    auxiliary_model.eval()

    model = deT5ForConditionalGeneration.from_pretrained(args.model).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print('model is load...')

    input_path = args.input_file
    print('data is load...')

    if args.compression:
        chunk_size = 1024
    else:
        chunk_size = 2048

    d = MultiEncoderDataset(
        args=args,
        data_path=input_path,
        tokenizer=auxiliary_tokenizer,
        chunk_size=chunk_size,
        num_passage=args.num_passage
    )
    dataloader = DataLoader(d, batch_size=args.batch_size, shuffle=False, collate_fn=d.my_collate)

    accuracy = []
    prompt_k = prompt_k * 2

    for idd, batch in enumerate(tqdm(dataloader)):
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        if args.compression == True:
            with torch.no_grad():
                # T5 encoder to decoder
                decoder_input_ids = torch.zeros((batch['input_ids'].shape[0], 1), dtype=torch.long)
                items_out = auxiliary_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    decoder_input_ids=decoder_input_ids.cuda(),
                    output_hidden_states=True,
                    return_dict=True,
                    use_knowledge_prompt=True
                )
            hidden = items_out.encoder_last_hidden_state
            gist_hidden = hidden[:, :prompt_k, :]

            now_gist_hidden = gist_hidden.reshape(int(gist_hidden.size()[0] / args.num_passage),
                                                  prompt_k * args.num_passage, -1)

            with torch.no_grad():
                raw_query = batch['raw_query']
                inpts = auxiliary_tokenizer(raw_query, return_tensors="pt", padding="longest", truncation=True).to(device)

                # T5 encoder to decoder
                emb = model.get_input_embeddings()
                input_embed = emb(inpts.input_ids)

                merged_embed = torch.cat((now_gist_hidden.to(device), input_embed), dim=1)
                decoder_input_tensor = now_gist_hidden.to(device)
                fist_dim = merged_embed.shape[0]
                second_dim = now_gist_hidden.shape[1]
                cross_attention_mask = torch.cat((torch.ones((fist_dim, second_dim)).to(device), inpts.attention_mask),
                                                 dim=1)

                gen = model.generate(input_ids=inpts.input_ids, attention_mask=inpts.attention_mask,
                                     max_new_tokens=args.max_new_tokens,
                                     decoder_input_tensor=decoder_input_tensor,
                                     cross_attention_mask=cross_attention_mask, )

        else:
            gen = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],max_new_tokens=args.max_new_tokens,)


        preds = []
        for ids in range(gen.shape[0]):
            pred = tokenizer.decode(gen[ids], skip_special_tokens=True)
            preds.append(pred)

        possible_answers = batch['labels']

        for iddd, poss_answer in enumerate(possible_answers):

            if any(isinstance(element, dict) for element in possible_answers[iddd]):
                possible_answers[iddd] = get_gold_answers(possible_answers[iddd])
            is_correct = False
            for pa in possible_answers[iddd]:
                if pa in preds[iddd] or pa.lower() in preds[iddd] or pa.capitalize() in preds[iddd]:
                    is_correct = True
            accuracy.append(is_correct)

        if idd % 100 == 0:
            print(sum(accuracy) / len(accuracy))


    print(sum(accuracy) / len(accuracy))
    print(args_dict)


if __name__ == "__main__":
    main()