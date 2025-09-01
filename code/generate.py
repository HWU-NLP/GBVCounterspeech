import unsloth
import argparse
import os
from pathlib import Path
from typing import Any, Dict
import yaml
import time
import pandas as pd
from tqdm import tqdm
from loguru import logger
import random
import numpy as np

import torch
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
from datasets import Dataset

from utils.processing import save_json
from dataset import GBV_MTConan, PreprocessedData
from utils.sampler import UniqueBatchSampler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_from_dataset_batch(
    dataset: PreprocessedData, 
    model: Any, 
    tokenizer: Any, 
    params: GenerationConfig,
    batch_size: int,
    random_state: int,
    device: torch.device
    ) -> None:

    inputs = dataset.tokenized_data()
    print('\n--> sampling')
    sampler = UniqueBatchSampler(
        labels=inputs["label"],
        group_keys=inputs["unique_labels"], 
        batch_size=batch_size, 
        random_state=random_state,
        mode="generate") 
   
    decoded_outputs = []
    n = 0
    for batch_indices in tqdm(sampler):
        batch = inputs.select(batch_indices)

        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=params,
            )
        decoded_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for idx, output in zip(batch["id"], decoded_output):
            decoded_outputs.append({'text': output,
                                    'prompt': output.split('Output:')[0], 
                                    'gbv': dataset.data['gbv_text'][idx],
                                    'cs_output': output.split('Output:')[1], 
                                    'gbv_id': int(dataset.data['gbv_id'][idx]),  
                                    'pair_id': int(dataset.data['pair_id'][idx]),
                                    'cs_human': dataset.data['counterspeech'][idx]})
          
    logger.warning(f"Number of outputs with errors: {n}\n")
    return decoded_outputs

def main():
    parser = argparse.ArgumentParser(description="Generate counterspeech given a file or in interative mode.")
    parser.add_argument("--config_file", required=True, help="Path to the config file")
    args = parser.parse_args()
    
    set_seed(seed=42)
    start_time = time.time()
    # Check if the config file exists
    assert os.path.exists(args.config_file), f"Config file '{args.config_file}' does not exist."
    
    with open(args.config_file, "r") as file:

        config = yaml.safe_load(file)
        device = config['device'] # for GPU usage or "cpu" for CPU usage
        use_unsloth = config['use_unsloth']
        
        if use_unsloth:
            print("\n#### Loading model with unsloth ####\n")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = config['model_name'], 
                max_seq_length = config['max_seq_length'],
                dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
                load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
            )
            FastLanguageModel.for_inference(model) # Enable native 2x faster inference
        else:
            tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
            #for llama
            tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
            tokenizer.pad_token_id = 0  # unk
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2
            # tokenizer.padding_side = "left"
            model = AutoModelForCausalLM.from_pretrained(config['model_name']).to(device)

        generation_params = GenerationConfig(
                    max_new_tokens=config['params']['max_new_tokens'], 
                    do_sample=True, 
                    temperature=config['params']['temperature'],
                    top_p=config['params']['top_p'],
                    pad_token_id=tokenizer.eos_token_id, 
        ) 
        
        # Check if the input file exists
        if os.path.exists(config['input_file']):
            dataset = GBV_MTConan(
                file=config['input_file'], 
                tokenizer=tokenizer,
                max_length=config['max_seq_length'],
                task= config['task'], # default = 'generate',
                label=['gbv_text'],
                labels_map={},
                task_prompt=config['task_prompt'],
                text_columns=config['text_columns'],
                batch_size=config['batch_size'],
                unsloth=use_unsloth,
                instruct = config['use_instruct'],
                chat_template = config['chat_template'],
                device=device,
            )
            print('\n--> generation')
            outputs = generate_from_dataset_batch(
                dataset=dataset, 
                model=model,
                tokenizer=dataset.tokenizer,
                params=generation_params,
                batch_size=config['batch_size'],
                random_state=config['random_state'],
                device=device
            )
            
            if config['model_name'].split('/')[-1].split('-')[0] == 'checkpoints':
                fpath = Path(config['output_file'], f"mtconan_{config['task_prompt']}_{config['model_name'].split('/')[-1].replace('/', '_')}.json")
            else:
                fpath = Path(config['output_file'], f"mtconan_{config['task_prompt']}_{config['model_name'].replace('/', '_')}.json")
            save_json(data=outputs, fpath=str(fpath))     
            logger.success(f"Generated {len(outputs)} outputs are save to {fpath}")
        else:
            config_fp = config['input_file']
            logger.error(f"Input file '{config_fp}' does not exist.")

    end_time = time.time()
    logger.info(f"Time cost: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()

"""
Sample usage
python generate.py --config_file=configurations/generate-mtconan.yaml
"""