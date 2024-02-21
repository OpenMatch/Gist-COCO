import argparse
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,trainer,OPTForCausalLM,AutoModelForCausalLM
from tqdm import tqdm
import torch
import numpy as np

import nltk
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


import datetime
import json
import math
from concurrent.futures.process import ProcessPoolExecutor
import jsonlines
import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizerBase,LlamaTokenizer,LlamaForCausalLM
from torch.utils.data import DataLoader
import evaluate

# self.new_gist_other_templete = ("Contex: {instruction} \nInput: {input} \nAnswer:")
# self.new_gist_templete = ("Input:{input} \nAnswer:")
# self.compression_qa = ("Context: {source} Input: {input}")
def postprocess_text(preds, labels, remove_llama_padding=False):
    if remove_llama_padding:
        # XXX: This is a temporary hack because skip_special_tokens doesn't
        # seem to be working with the Llama SentencePiece tokenizer?
        preds = [pred.replace("⁇", "") for pred in preds]
        labels = [pred.replace("⁇", "") for pred in labels]

    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

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
        data_type=None,
        aug_type = True,
        num_passage =1,
    ):

        self.tokenizer = tokenizer
        self.chunk_size=chunk_size
        self.data_type = data_type
        self.num_passage  =num_passage

        self.num_samples = num_samples
        self.verbose = verbose
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.aug_type=aug_type

        self.instruction_other_templete = ("Context: {instruction} \nInput: {input} \nAnswer:")
        self.instruction_templete = ("Input:{input} \nAnswer:")
        self.compression_qa = ("Context: {source} Input: {input}")
        self.qa_templete ="In this task, you're given a question. Your task is to provide a short answer to the given question. Question:{} Answer:"
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

        output = self.tokenizer(now_passage_list, return_tensors="pt", max_length=self.chunk_size,
                                padding="longest", truncation=True)
        input_ids = output['input_ids']
        attention_mask = output['attention_mask']

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'answers': labels, 'raw_query': raw_query,
                'passages': passages}

    def _encode_data(self, file_path, max_workers):
        if self.data_type == 'instruction':
            with jsonlines.open(file_path) as reader:
                for line in reader:
                    file_path = line
            encodings = list(map(self._process_line, enumerate(file_path)))
        else:
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
        if self.data_type == 'instruction':
           data = line
        else:
            data = json.loads(line)

        if self.data_type == "passage":
            if 'output' in data:
                query = self.qa_templete.format(data['question'])
            else:
                #POPQA dataset has it's official templete: "Q: {} A:":
                query = data['prompt']
            if 'output' in data:
                target = data['output']
            else:
                target = data['answers']
            source = ' '.join(data['passages'][:self.num_passage])
            passages = data['passages']

        elif self.data_type == "instruction":
            query =data['input']
            target = data['output']
            source = data['instruction']
            passages = data['instruction']


        else:
            if any(isinstance(element, dict) for element in data['labels']):
                query = self.qa_templete.format(data['prompts'])
            elif any(isinstance(element, dict) for element in data['passages']):
                query = self.qa_templete.format(data['prompts'])
            else:
                query = data['prompts']
            source = data['compression_passage']
            target = data['labels']
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

    def _encode_example(self, source, target,passages=None, query=None):

        if self.aug_type:
            if self.data_type =='instruction_compress':
                now_passage = self.instruction_other_templete.format_map(dict(instruction=source,input=query))
            elif self.data_type =='instruction':
                now_passage = self.instruction_other_templete.format_map(dict(instruction=source, input=query))
            elif self.data_type =='passage_compress':
                   now_passage = self.compression_qa.format_map(dict(source=source, input=query))
            else:
                   now_passage = source + ' ' + query

        else:
            if self.data_type =='instruction':
                now_passage = self.instruction_templete.format_map(dict(input=query))

            else:
                now_passage = query




        return {
            'now_passage': now_passage ,
            'labels': target,
            'passages': passages,
            'raw_query': query
        }


def call_model(input_ids, attention_mask, model, tokenizer, device,args):

    if 'instruction' in args.data_type:
        max_new_tokens = 32

    else:
        max_new_tokens = 15
    gen = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
    preds = []
    texts = []

    for ids, tokens in enumerate(input_ids):
        text = tokenizer.decode(gen[ids], skip_special_tokens=True)
        actual_prompt = tokenizer.decode(tokens, skip_special_tokens=True)
        pred = text[len(actual_prompt):]
        if pred.startswith("\n\n"):
            pred = pred[2:]
        pred = pred.split("\n")[0]
        preds.append(pred)
        texts.append(text)
    return preds, texts


def get_gold_answers(gold):
    ground_truths = set()
    for item in gold:
        if "answer" in item and item["answer"] and len(item["answer"].strip()) > 0:
            ground_truths.add(item["answer"].strip())
    return ground_truths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='/data4/LLaMA/llama-7b')
    parser.add_argument('--input_file', type=str,default="/data3/lixinze/compression_model/task_knowledge_prompt/ACL_compression/open_code_check/unseen.json")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--data_type', type=str, default="instruction_compress")
    parser.add_argument('--num_passage', type=int, default=5)

    parser.add_argument('--augment_type', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)


    args = parser.parse_args()
    args_dict = vars(args)
    print(args_dict)
    batch_size = args.batch_size


    model_name = args.model_name
    device = args.device


    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = '[PAD]'
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_name).eval().to(device)



    print('model is load...')
    generate = lambda input_ids, attention_mask,: call_model(input_ids, attention_mask, model=model, tokenizer=tokenizer, device=device,args=args,)

    input_path = args.input_file
    print(input_path)
    print('data is load...')

    chunk_size = 2048
    d = MultiEncoderDataset(
        data_path=input_path,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        data_type=args.data_type,
        aug_type = args.augment_type,
        num_passage=args.num_passage
    )


    device = "cuda"
    my_collate=d.my_collate
    dataloader = DataLoader(d, batch_size=batch_size, shuffle=False, collate_fn=my_collate)


    accuracy = []
    all_pred = []
    all_labels = []
    rougeL = []
    results = {}

    if 'instruction' not in args.data_type:
        for idd, batch in enumerate(tqdm(dataloader)):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            preds, response = generate(input_ids, attention_mask,)
            possible_answers = batch['answers']

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

    else:
        for idd, batch in enumerate(tqdm(dataloader)):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            preds, response = generate(input_ids, attention_mask)
            possible_answers = batch['answers']


            decoded_preds, decoded_labels = postprocess_text(
                preds,
                possible_answers,
            )

            all_pred.append(decoded_preds)
            all_labels.append(decoded_labels)

        all_pred = sum(all_pred, [])
        all_labels = sum(all_labels, [])

        rouge_results = evaluate.load(
            "/data1/lixinze/gist/ours/task_knowledge_prompt/decoder_generation/gisting/evaluate-main/evaluate-main/metrics/rouge").compute(
            predictions=all_pred, references=all_labels, use_stemmer=True
        )
        rouge_results = {k: round(v * 100, 4) for k, v in rouge_results.items()}
        results.update(rouge_results)

        rougeL.append(results['rougeL'])
        print(results)
        print(sum(rougeL) / len(rougeL))
    print(args_dict)



if __name__ == "__main__":
    main()