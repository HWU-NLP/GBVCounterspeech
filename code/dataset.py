import pandas as pd
import re
from ast import literal_eval
from loguru import logger

from datasets import Dataset
import torch
        
from utils.prompts import (
    system_prompt,
    llama3_chat_template,
    phi4_chat_template,
    cs_generation_baseline_0, 
    cs_generation_0_1, 
    cs_generation_0_2, 
    cs_generation_form_0, 
    cs_generation_form_target_0, 
)

class PreprocessedData:
    def __init__(
        self, instruct, unsloth, chat_template, file, tokenizer, max_length, task, label, 
        labels_map, task_prompt, text_columns, batch_size, device
    ) -> None:
        super(PreprocessedData, self).__init__()
        self.instruct = instruct
        self.unsloth = unsloth
        self.chat_template = chat_template
        self.system_prompt = system_prompt
        self.device = device
        
        self.file = file
        self.task = task 
        self.label_column = label
        self.multilabel = True if task == 'classify_bert' else False
        self.data, self.processed_label_column, self.labels = self.read_file() 
        self.data_size = self.data.shape[0]
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.batched = False if batch_size <= 0 else True 
        self.max_length = max_length
        self.labels_map = labels_map if len(labels_map) > 1 else {}
        self.task_prompt = task_prompt
        self.text_column = text_columns
        self.prompt_creation()
        self.labels_encoded()
        self.input_encodings = self.calculate_encodings_input() if self.text_column else "Provide text column if you need it"
        self.output_encodings = self.calculate_encodings_label() if len(self.processed_label_column) == 1 else None

        logger.info(f"\n\n******************* Dataset Stats *******************\n \
                    Total dataset: {self.data_size}\n \
                    Total unique labels: {len(self.labels)}\n \
                    ")
        
    def read_file(self):
        pass
    
    #for bert: binary/multiclass (txt1 : 2, txt2: 1, txt3: 1, ..)
    def transform_label_col(self): 
        if not self.labels_map:
            for l in self.labels:
                if l not in self.labels_map:
                    self.labels_map[l] = len(self.labels_map)
        self.data["label_encoded"] = self.data[self.processed_label_column[0]].apply(
            lambda x: self.labels_map[x])        
    
    # for bert: multilabel (txt1 : 0,1; txt2: 0,0; txt3: 1,1; ..)
    def transform_multilabel_col(self): 
        list_labels = []
        for i in range(self.data_size):
            i_labels = []
            for l in self.processed_label_column:
                i_labels.append(int(self.data[l][i]))
            list_labels.append(i_labels)
        self.data["label_encoded"] = list_labels

    def labels_encoded(self): 
        if self.task == 'classify_bert':
            if self.multilabel:
                if len(self.processed_label_column) > 2:
                    # array of arrays
                    self.transform_multilabel_col()
                else:
                    # array
                    self.transform_label_col() 
        else:
            # self.output_encodings
            self.calculate_encodings_label()

    def apply_instruction_template(self, input, output=None):
        # input is going to be a full prompt with an instruction at the preamble.       
        if self.instruct:
            # self.tokenizer.pad_token = self.tokenizer.eos_token 
            self.tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
            self.tokenizer.pad_token_id = 0  # unk
            self.tokenizer.bos_token_id = 1
            self.tokenizer.eos_token_id = 2
            
            chat_templates = {
                'llama-32': llama3_chat_template,
                'phi-4': phi4_chat_template,
            }

            if self.unsloth:
                # chat template needed for unsloth according the standards
                chat = [{'content': input, 'role': 'user'}] 

                if self.system_prompt == None:
                    print('***ERROR: Set a system prompt')
                else:
                    if output is not None:
                        chat = [
                            {'content': self.system_prompt,'role': 'system'},
                            {'content': input, 'role': 'user'}, 
                            {'content': output, 'role': 'assistant'},
                        ] 
                        chat_template = chat_templates[self.chat_template][1]
                        chat_template = re.sub(r'{SYSTEM}', self.system_prompt, chat_template)
                        chat_template = re.sub(r'{INPUT}', input, chat_template)
                        chat_template = re.sub(r'{OUTPUT}', output, chat_template)
                    else:
                        chat = [
                            {'content': self.system_prompt,'role': 'system'},
                            {'content': input, 'role': 'user'}, 
                        ] 
                        chat_template = chat_templates[self.chat_template][0]
                        chat_template = re.sub(r'{SYSTEM}', self.system_prompt, chat_template)
                        chat_template = re.sub(r'{INPUT}', input, chat_template)
                    return self.tokenizer.apply_chat_template(chat, tokenize=False, chat_template=chat_template, add_generation_prompt=True)
                    
            else:
                return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        else:
            return input
        
    def prompt_creation(self):
        if self.task_prompt != '':
            print('\n used prompt: ', self.task_prompt)
            print('\n text columns: ', self.text_column[0], '\n')

            tasks = {
                'simple-zero-shot-baseline': cs_generation_baseline_0,
                'simple-zero-shot-1' : cs_generation_0_1,
                'simple-zero-shot-2': cs_generation_0_2,
                'form-zero-shot': cs_generation_form_0, 
                'target-zero-shot': cs_generation_form_target_0, 
                }

            if len(self.text_column) == 1:
                self.data['prompt'] = self.data.apply(
                        lambda x: self.apply_instruction_template(
                            tasks[self.task_prompt](x[self.text_column[0]])
                        ), 
                        axis=1
                    )
                self.data['prompt_and_label'] = self.data.apply(
                        lambda x: self.apply_instruction_template(
                            tasks[self.task_prompt](x[self.text_column[0]]),
                            x['counterspeech']
                        ), 
                        axis=1
                    )
                self.data['raw_prompt'] = self.data.apply(
                        lambda x: tasks[self.task_prompt](x[self.text_column[0]]), 
                        axis=1
                    )
            elif len(self.text_column) == 2:
                self.data['prompt'] = self.data.apply(
                        lambda x: self.apply_instruction_template(
                            tasks[self.task_prompt](x[self.text_column[0]], 
                                                    x[self.text_column[1]])
                        ), 
                        axis=1
                    )
                self.data['prompt_and_label'] = self.data.apply(
                        lambda x: self.apply_instruction_template(
                            tasks[self.task_prompt](x[self.text_column[0]], 
                                                    x[self.text_column[1]]),
                            x['counterspeech']
                        ), 
                        axis=1
                    )
                self.data['raw_prompt'] = self.data.apply(
                        lambda x: tasks[self.task_prompt](x[self.text_column[0]], 
                                                    x[self.text_column[1]]), 
                        axis=1
                    )
            elif len(self.text_column) == 3:
                self.data['prompt'] = self.data.apply(
                        lambda x: self.apply_instruction_template(
                            tasks[self.task_prompt](x[self.text_column[0]], 
                                                        x[self.text_column[1]],
                                                        x[self.text_column[2]])
                        ), 
                        axis=1
                    )
                self.data['prompt_and_label'] = self.data.apply(
                        lambda x: self.apply_instruction_template(
                            tasks[self.task_prompt](x[self.text_column[0]], 
                                                        x[self.text_column[1]],
                                                        x[self.text_column[2]]),
                            x['counterspeech']
                        ), 
                        axis=1
                    )
                self.data['raw_prompt'] = self.data.apply(
                        lambda x: tasks[self.task_prompt](x[self.text_column[0]], 
                                                    x[self.text_column[1]],
                                                    x[self.text_column[2]]), 
                        axis=1
                    )
            elif len(self.text_column) == 4:
                self.data['prompt'] = self.data.apply(
                        lambda x: self.apply_instruction_template(
                            tasks[self.task_prompt](x[self.text_column[0]], 
                                                        x[self.text_column[1]],
                                                        x[self.text_column[2]],
                                                        x[self.text_column[3]],
                                                        )
                        ), 
                        axis=1
                    )
                self.data['prompt_and_label'] = self.data.apply(
                        lambda x: self.apply_instruction_template(
                            tasks[self.task_prompt](x[self.text_column[0]], 
                                                        x[self.text_column[1]],
                                                        x[self.text_column[2]],
                                                        x[self.text_column[3]],
                                                        ),
                            x['counterspeech']
                        ), 
                        axis=1
                    )
                self.data['raw_prompt'] = self.data.apply(
                        lambda x: tasks[self.task_prompt](x[self.text_column[0]], 
                                                    x[self.text_column[1]],
                                                    x[self.text_column[2]],
                                                    x[self.text_column[3]]), 
                        axis=1
                    )
            else:
                print('Implementation needed for more than 4 columns in the prompt')

            print('\nexample of input with prompt:\n')
            print(self.data['prompt'][10])

            print('\nexample of input with prompt and label:\n')
            print(self.data['prompt_and_label'][10])
        
        else:
            pass

    def calculate_encodings_input(self): 
        if self.task == "evaluate_generation":
            # for evaluate generation task, encode prompt+label text 
            encodings = self.tokenizer(
                self.input_label_texts(),
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
        else:
            encodings = self.tokenizer(
                self.input_texts(),
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
        return encodings

    def calculate_encodings_label(self):
        # e.g., '1 2 3' or 'gbv' 
        if self.task == "evaluate_generation":
            # for evaluate generation task, encode prompt+label text but ignore index before label
            # make input tokens to -100 so that the model doesn't compute loss on them
            input_label_encodings = self.tokenizer(
                self.input_label_texts(),
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to('cuda')
            input_encodings = self.tokenizer(
                self.input_texts(),
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to('cuda')
            input_label_ids = input_label_encodings["input_ids"]
            input_ids = input_encodings["input_ids"]
            label_encodings = input_label_ids.clone()
            mask = (input_label_ids == input_ids) & (input_ids != self.tokenizer.pad_token_id)
            label_encodings[mask] = -100

        else:
            label_encodings = self.tokenizer(
                self.label_text(), 
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to('cuda')
        return label_encodings

    def input_texts(self):
        if self.task == "classify_bert":
            return self.data[self.text_column[0]].values.tolist()
        else:
            # generation
            return self.data["prompt"].values.tolist() 

    def label_text(self): 
        # generation
        return self.data['label_text'].values.tolist()
    
    def input_label_texts(self):
        # evaluate_generation
        return self.data["prompt_and_label"].values.tolist()

    def __getitem__(self, idx):
        if self.task == 'classify_bert':
            input_ids = self.input_encodings["input_ids"][idx]
            attention_mask = self.input_encodings["attention_mask"][idx]
            # token_type_ids = self.input_encodings["token_type_ids"][idx]
            label = torch.tensor(self.data["label_encoded"][idx], dtype=torch.float32)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                # "token_type_ids": token_type_ids,
                "labels": label,
            }
        elif self.task == 'generate':
            input_ids = self.input_encodings["input_ids"][idx]
            attention_mask = self.input_encodings["attention_mask"][idx]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        elif self.task == 'evaluate_generation':
            # only for a training purpose
            input_ids = self.input_encodings["input_ids"][idx]
            attention_mask = self.input_encodings["attention_mask"][idx]
            label = self.output_encodings[idx] 
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": label,
            }
        else:
            # evaluate generation and classify (llm): both input and output are strings
            input_ids = self.input_encodings["input_ids"][idx]
            attention_mask = self.input_encodings["attention_mask"][idx]
            #only for a training purpose:
            label = self.output_encodings["input_ids"][idx] 
            # for classify we use a sentence of strategies converted in 1 2 3... 
            # in order to get single tokens
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": label,
            }

    def encoding_fn(self, sample):
        if type(sample) == int:
            return self.__getitem__(sample)   
        else:
            for idx in sample:
                return self.__getitem__(idx)

    def __len__(self):
        return self.data_size

    def __num_labels__(self):
        return len(self.labels)

    def __data__(self):
        return self.data


class GBV_MTConan(PreprocessedData):
 
    def read_file(self):
        data = pd.read_csv(self.file, sep='\t', header=0)
        labels_dict = {}
        label = self.label_column[0]
        print('label column: ', label)

        # reading columns with texts and not lists of labels, e.g., 'gbv_text'
        if '[' not in data[label].tolist()[1]:                 
            data['label_text'] = data[label].tolist()
            processed_label_column = ['label_text']
            labels = data[label].unique()
            return data, processed_label_column, labels

        else:
            # reading columns with lists of labels, e.g., 'gbv_form'
            for e, c in enumerate(data[label].tolist()):
                c = literal_eval(c)
                for i in c:
                    if ',' in i:
                        list_i = i.split(',')
                        list_i = [x.strip() for x in list_i] 
                        for x in list_i:       
                            if e in labels_dict.keys():
                                labels_dict[e].update({x: c.count(x)})
                            else:
                                labels_dict[e]={x: c.count(x)}
                    else:
                        if e in labels_dict.keys():
                            labels_dict[e].update({i: c.count(i)})
                        else:
                            labels_dict[e]={i: c.count(i)}
            # print(labels_dict)
            labels = []
            for k, v in labels_dict.items():
                for v1, v2 in v.items():
                    if v1 not in labels:
                        labels.append(v1)
            labels.sort(reverse=False)

            if self.multilabel:  
                # classify_bert    
                for k,v in labels_dict.items():
                    for l in labels:
                        if l in v.keys():
                            data.at[k, l] = 1 
                        else:
                            data.at[k, l] = 0
                processed_label_column = labels
                print("list of labels' columns: ", processed_label_column)
                return data, processed_label_column, labels
            
            else: 
                # generation of labels 
                for k,v in labels_dict.items():
                    list_labels = []
                    for v1, v2 in v.items():
                        if v1 in labels:
                            list_labels.append(f'{labels.index(v1)+1}')
                    #'1 2 3'
                    data.at[k, 'label_text'] = ' '.join(list_labels) 

                processed_label_column = ['label_text']
                # print('processed_label_column: ', data['label_text'])
                return data, processed_label_column, labels

    def tokenized_data(self):
        list_idx = []
        if self.multilabel:
            label_list = self.data['label_encoded'].values.tolist()
        else:
            label_list=self.label_text()

        list_idx = self.data.index.tolist()
        texts = self.input_texts()
        if self.task != 'classify_bert':
            dataset = {'id': list_idx, 'text': texts, 'label': label_list, 'unique_labels': self.data['gbv_text'].tolist()}
        else:
            dataset = {'id': list_idx, 'text': texts, 'label': label_list}
        dataset = Dataset.from_dict(dataset)
        tokenized_dataset = dataset.map(lambda example: self.encoding_fn(example["id"]))
        return tokenized_dataset  # no batched

    def splitting(self, test_size, seed=32):
        tokenized_dataset = self.tokenized_data()
        if self.task == 'classify_bert':
            if not self.multilabel:
                dataset_split = tokenized_dataset.train_test_split(test_size=test_size, seed=seed, stratify_by_column='label')
                dataset_split = dataset_split.remove_columns(['id', 'text', 'label'])
            else:
                dataset_split = tokenized_dataset.train_test_split(test_size=test_size, seed=seed)
                dataset_split = dataset_split.remove_columns(['id', 'text', 'label'])
 
        else:
            dataset_split = tokenized_dataset.train_test_split(test_size=test_size, seed=seed)
        return dataset_split
    