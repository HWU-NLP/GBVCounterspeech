"""
Convert generated cs response to the preference data format based on reward scores.

see more details about preference format and examples here: 
https://huggingface.co/docs/trl/main/en/dataset_formats#preference

extract prompt from chosen/reject dict
https://huggingface.co/docs/trl/main/en/data_utils#trl.extract_prompt
"""

import os
import yaml
import argparse
import pandas as pd
from random import choice
import re

from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel

from dataset import GBV_MTConan
from utils.processing import save_json, clean_cs
from utils.prompts import system_prompt, llama3_chat_template

CS_COLUMNS = [
    'cs_prompting_0',
    'cs_strategy_single',
    'cs_strategy_multi',
    'cs_strategy_form',
    'cs_strategy_form_target'
]

TASKS = [
    'simple-zero-shot-baseline',
    'simple-zero-shot-1',
    'simple-zero-shot-2',
    'form-zero-shot', 
    'target-zero-shot', 
]

TEXT_COLUMNS = [
    ['gbv_text'],
    ['gbv_text', 'cs_strategy'],
    ['gbv_text', 'cs_strategy'],
    ['gbv_text', 'cs_strategy', 'gbv_form'],
    ['gbv_text', 'cs_strategy', 'gbv_form', 'gbv_target']
]

def load_gbv_dataset(tokenizer, config):
    datasets = []
    for idx in range(5):
        dataset = GBV_MTConan(
                    file=config['input_file'], 
                    tokenizer=tokenizer,
                    max_length=config['max_length'],
                    task='generate',
                    label=config['label'],
                    labels_map={},
                    task_prompt=TASKS[idx],
                    text_columns=TEXT_COLUMNS[idx],  
                    batch_size=1,
                    instruct = config['use_instruct'],
                    unsloth = config['use_unsloth'],
                    chat_template = config['chat_template'],
                    device = config['device'],
                )
        df = dataset.data.set_index("pair_id")
        datasets.append(df)
    return datasets


def select_chosen_and_rejected(df, datasets):
    examples = []

    for _, row in df.iterrows():
        pair_id = row['pair_id']
        gbv = row['gbv']
        answers = eval(row['answer']) 

        for annotator_answer in answers:
            cs_feedback = annotator_answer.split('@@@')
            
            cs_grouped = []
            for idx, fb in enumerate(cs_feedback):
                gbv_df = datasets[idx]

                try:
                    prompt = gbv_df.loc[pair_id]['raw_prompt']
                except KeyError:
                    continue

                cs_text = row[CS_COLUMNS[idx]]
                if not isinstance(cs_text, str):
                    continue

                try:
                    q1, q2 = eval(fb)
                except:
                    continue

                cs_grouped.append({
                    "prompt": prompt,
                    "gbv": gbv,
                    "cs": clean_cs(cs_text),
                    "feedback": (q1, q2),
                })

            preferred = [g for g in cs_grouped if g['feedback'] == (1, 1)]
            rejected = [g for g in cs_grouped if g['feedback'] != (1, 1)]

            for pos in preferred:
                for neg in rejected:
                    examples.append({
                        "system": [{'content': system_prompt,'role': 'system'}],
                        "prompt": [{"role": "user", "content": pos['prompt']}],
                        "chosen": [{"role": "assistant", "content": pos['cs']}],
                        "rejected": [{"role": "assistant", "content": neg['cs']}]
                    })
    return examples

def apply_chat_templates(examples, tokenizer):
    if all(k in examples[0].keys() for k in ("chosen", "rejected")):
        chat_examples = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }
        
        for example in examples:
            chat_prompt = [example["system"][0], example["prompt"][0]]
            chat_chosen = [example["system"][0], example["prompt"][0], example["chosen"][0]]
            chat_rejected = [example["system"][0], example["prompt"][0], example["rejected"][0]]
            
            chat_template = llama3_chat_template
            
            chat_template_prompt = re.sub(r'{SYSTEM}', system_prompt, chat_template[0])
            chat_template_prompt = re.sub(r'{INPUT}', example["prompt"][0]["content"], chat_template_prompt)
            
            chat_template_chosen = re.sub(r'{OUTPUT}', example["chosen"][0]["content"], chat_template[2])
            chat_template_rejected = re.sub(r'{OUTPUT}', example["rejected"][0]["content"], chat_template[2])
            
            chat_examples["prompt"].append(tokenizer.apply_chat_template(chat_prompt, tokenize=False, chat_template=chat_template_prompt, add_generation_prompt=True))
            chat_examples["chosen"].append(tokenizer.apply_chat_template(chat_chosen, tokenize=False, chat_template=chat_template_chosen, add_generation_prompt=True))
            chat_examples["rejected"].append(tokenizer.apply_chat_template(chat_rejected, tokenize=False, chat_template=chat_template_rejected, add_generation_prompt=True))
    
    else:
        raise ValueError(
            f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    
    return Dataset.from_dict(chat_examples)
                
def main():
    parser = argparse.ArgumentParser(description="Preference Dataset Conversion Configuration")
    parser.add_argument("--config_file", required=True, help="Path to the config file")
    args = parser.parse_args()
    
    # Check if the config file exists
    assert os.path.exists(args.config_file), f"Config file '{args.config_file}' does not exist."
    
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    output_folder = config["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    else:
        print(f"Output folder exists: {output_folder}")

    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config['model_name'],
        max_seq_length = config['max_length'],
        dtype = None, 
        load_in_4bit = True, 
    )
    tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
    tokenizer.pad_token_id = 0  # unk
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    datasets = load_gbv_dataset(tokenizer, config)
    
    # format preference dataset
    reward_df = pd.read_csv(config["reward_dataset"], sep="\t")
    prefer_data_list = select_chosen_and_rejected(reward_df, datasets)
    save_fp = os.path.join(output_folder, "preference_data.json")
    save_json(prefer_data_list, save_fp)

    print(f"Generated {len(prefer_data_list)} preference pairs and saved to {save_fp}")

    # convert to huggingface dataset and apply chat template
    prefer_dataset = apply_chat_templates(prefer_data_list, tokenizer)
    prefer_dataset = prefer_dataset.train_test_split(test_size=config["TEST_SIZE"])
    
    save_hf_fp = os.path.join(output_folder, "preference_data_hf")
    prefer_dataset.save_to_disk(save_hf_fp)
    print(f"Saved huggingface preference data to {save_hf_fp}")

    
if __name__ == "__main__":
    main()

"""
Sample usage
python preference.py --config_file=configurations/preference-mtconan.yaml
"""