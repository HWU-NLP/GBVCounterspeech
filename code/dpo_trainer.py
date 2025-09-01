# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb#scrollTo=AqkY_wHdKyOl

import os
import argparse
import yaml
import random
import time
import numpy as np
from loguru import logger
from pathlib import Path
from copy import deepcopy

from datasets import load_dataset, load_from_disk
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth import PatchDPOTrainer
PatchDPOTrainer()

from trl import DPOTrainer, DPOConfig
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers.integrations import NeptuneCallback

from huggingface_hub import login

neptune_api_token = "<your_neptune_api_token>"
neptune_project = "<your_neptune_project_name>"

hub_token = "<your_huggingface_hub_token>"
login(token=hub_token)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False

def print_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

def main():
    parser = argparse.ArgumentParser(description="DPO Training Configuration")
    parser.add_argument("--config_file", required=True, help="Path to the config file")
    args = parser.parse_args()
    
    # Check if the config file exists
    assert os.path.exists(args.config_file), f"Config file '{args.config_file}' does not exist."
    
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    set_seed(config['RANDOM_STATE'])
    
    output_folder = config["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Created output folder: {output_folder}")
    else:
        logger.info(f"Output folder exists: {output_folder}")

    dash_line = "\n" + "_".join(" " for x in range(60))

    logger.info(dash_line)
    num_epochs = config['training_config']['num_epochs']
    learning_rate = config['training_config']['learning_rate']
    batch_size = config['training_config']['batch_size']
    max_length = config['training_config']['max_length']
    max_prompt_length = config['training_config']['max_prompt_length']
    beta = config['training_config']['beta']
    model_name = config['training_config']['model_name']    
    logger.info(f"Training configuration:")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max length: {max_length}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Output folder: {output_folder}")
    
    
    logger.info(dash_line)
    logger.info(f"**** Loading model: {model_name}\n")
    
    device = config['device'] 
    use_unsloth = config['use_unsloth']
    if use_unsloth:
        logger.info("\n#### Loading model with unsloth ####\n")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name, 
            max_seq_length = max_length,
            dtype = None, 
            load_in_4bit = True, 
        )
        tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
        tokenizer.pad_token_id = 0  # unk
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        
        model = FastLanguageModel.get_peft_model(
            model,
            r = config["lora_config"]["r"], 
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = config["lora_config"]["lora_alpha"],
            lora_dropout = config["lora_config"]["lora_dropout"],
            bias = "none",    
            use_gradient_checkpointing = "unsloth", 
            random_state = config['RANDOM_STATE'],
            use_rslora = False,  
            loftq_config = None, 
        )
        
    else:
        logger.info("\n#### Loading model with transformers ####\n")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            load_in_4bit=True, 
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token_token
        tokenizer.padding_side = "left"  
        model.resize_token_embeddings(len(tokenizer)) 
        model = get_peft_model(
            model,
            r = config["lora_config"]["r"], 
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = config["lora_config"]["lora_alpha"],
            lora_dropout = config["lora_config"]["lora_dropout"], 
            bias = "none",    
    
    prefer_dataset = load_from_disk(config["input_file"])
    train_dataset = prefer_dataset['train']
    test_dataset = prefer_dataset['test']

    logger.info(f"Model parameters:\n{print_trainable_model_parameters(model)}\n")
    
    logger.info(dash_line)
    logger.info(f"**** Starting DPO Training\n")
    output_checkpoint = f"checkpoints-{model_name.replace('/', '_')}-dpo"
    output_dir = Path(output_folder, output_checkpoint)
    neptune_run_name = f"{model_name}-dpo-{str(int(time.time()))}" 

    neptune_cb = NeptuneCallback(
        api_token=neptune_api_token,
        project=neptune_project,
        name=neptune_run_name,
    )
    
    early_stopping_cb = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.0
    )

    # Initialise the trainer
    dpo_config = DPOConfig(
        per_device_train_batch_size=batch_size,  
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = num_epochs,
        learning_rate = learning_rate,
        max_length = max_length,
        max_prompt_length = max_prompt_length,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        beta = beta,
        lr_scheduler_type = "linear",
        evaluation_strategy="epoch", 
        save_strategy="epoch",
        save_total_limit=1,  # keep only the best one and the most recent n-1 checkpoints
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model="eval_rewards/margins",  # "eval_" prefix used -> see get_batch_loss_metrics() metric names.
        seed = config['RANDOM_STATE'],
        output_dir = output_dir,
        report_to = "none", 
    )

    dpo_trainer = DPOTrainer(
        model = model,
        ref_model = None,
        args = dpo_config,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        tokenizer = tokenizer, 
        callbacks=[neptune_cb, early_stopping_cb],
    )
    
    dpo_trainer.train()

    logger.info(dash_line)
    logger.info(f"Saving finetuned model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()

"""
Sample usage
python dpo_trainer.py --config_file=configurations/dpo-training-mtconan.yaml
"""