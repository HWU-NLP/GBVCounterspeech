import pandas as pd
import numpy as np
import time
import torch
import argparse
import os
from pathlib import Path
from loguru import logger
import yaml
import random

from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding, EarlyStoppingCallback
from transformers.integrations import NeptuneCallback
from peft import LoraConfig, get_peft_model

from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from trl import SFTTrainer
from huggingface_hub import login

from dataset import GBV_MTConan
from utils.sampler import UniqueBatchSampler


neptune_api_token = "<your_neptune_api_token>"
neptune_project = "<your_neptune_project_name>"

hub_token = "<your_huggingface_hub_token>"
login(token=hub_token)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False
    
class DataCollatorForPrompt:
    def __init__(self, tokenizer):
        self.collator = DataCollatorWithPadding(tokenizer, padding=True)  # padding="max_length",

    def __call__(self, batch):
        # Only keep model inputs
        cleaned_batch = [{k: v for k, v in d.items() if k in ['input_ids', 'attention_mask', 'labels']} for d in batch]
        return self.collator(cleaned_batch)

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, random_state=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = random_state
        
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=UniqueBatchSampler(
                labels=[s['label'] for s in self.train_dataset],
                group_keys=[s['unique_labels'] for s in self.train_dataset],
                batch_size=self.args.per_device_train_batch_size,
                random_state=self.random_state,
            ),
            collate_fn=DataCollatorForPrompt(self.tokenizer),  
            num_workers=2
        )
        
    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_sampler=UniqueBatchSampler(
                labels=[s['label'] for s in self.eval_dataset],
                group_keys=[s['unique_labels'] for s in self.eval_dataset],
                batch_size=self.args.per_device_eval_batch_size,
                random_state=self.random_state,
            ),
            collate_fn=DataCollatorForPrompt(self.tokenizer), 
            num_workers=2
        )


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


def main():
    parser = argparse.ArgumentParser(description="Training llms with peft strategy")
    parser.add_argument("--config_file", required=True, help="Path to the config file with input/output paths and model configuration.")
    args = parser.parse_args()

    start_time = time.time()
    
    # Check if the config file exists
    assert os.path.exists(args.config_file), f"Config file '{args.config_file}' does not exist."    
    file = open(args.config_file, "r")
    config = yaml.safe_load(file)

    set_seed(config['RANDOM_STATE'])

    # Check if output folder exists, if not, create it
    output_folder = config['output_folder']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Created folder: {output_folder}")
    else:
        logger.info(f"Folder already exists: {output_folder}")
    
    dash_line = "\n" + "_".join(" " for x in range(60))

    logger.info(dash_line)
    # Print the training configuration
    num_epochs = config['training_config']['num_epochs']
    learning_rate = config['training_config']['learning_rate']
    batch_size = config['training_config']['batch_size']
    max_length = config['training_config']['max_length']
    model_name = config['training_config']['model_name']    
    logger.info(f"Training configuration:")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max length: {max_length}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Output folder: {output_folder}")

    dash_line = "\n" + "_".join(" " for x in range(100))

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

        peft_model = FastLanguageModel.get_peft_model(
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
        logger.debug(tokenizer.special_tokens_map)
        logger.debug(tokenizer.all_special_tokens)
        logger.debug(tokenizer.all_special_ids)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        #for llama
        tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
        tokenizer.pad_token_id = 0  # unk
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        # tokenizer.padding_side = "left"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # load_in_8bit=True,
        ).to(device)
    

        lora_config = LoraConfig(
            r=config["lora_config"]["r"],
            lora_alpha=config["lora_config"]["lora_alpha"],
            target_modules=["q_proj","v_proj"],
            lora_dropout=config["lora_config"]["lora_dropout"],
            task_type=config["lora_config"]["task_type"],
            bias="none",
        )

        peft_model = get_peft_model(model, lora_config)
    
    logger.info(print_number_of_trainable_model_parameters(peft_model))


    logger.info(dash_line)
    logger.info(f"**** Loading dataset from {config['input_file']}\n")
    dataset = GBV_MTConan(
        file=config['input_file'], 
        tokenizer=tokenizer,
        max_length=config['training_config']['max_length'],
        task=config['task'],
        label=config['label'],
        labels_map={},
        task_prompt=config['task_prompt'],
        text_columns=config['text_columns'],
        batch_size=batch_size,
        instruct = config['use_instruct'],
        unsloth = use_unsloth,
        chat_template = config['chat_template'],
        device=device,
    )
    
    split_dataset = dataset.splitting(test_size=config['TEST_SIZE'], seed=config['RANDOM_STATE'])
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    logger.debug(train_dataset)
    logger.debug(val_dataset)
    
    logger.info(dash_line)
    logger.info(f"**** Starting PEFT fine-tuning\n")
    output_checkpoint = f"checkpoints-{config['task_prompt']}-{model_name.replace('/', '_')}-sft" 
    output_dir = Path(output_folder, output_checkpoint)
    neptune_run_name = f"{config['task_prompt']}-{model_name}-sft-{str(int(time.time()))}"

    neptune_cb = NeptuneCallback(
        api_token=neptune_api_token,
        project=neptune_project,
        name=neptune_run_name,
    )
    
    early_stopping_cb = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.0
    )

    peft_training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,  # still specify, but will be controlled by BatchSampler
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs=num_epochs,
        learning_rate = learning_rate, 
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        evaluation_strategy="steps",  
        eval_steps=10,  
        metric_for_best_model="eval_loss",
        save_strategy="steps",
        save_steps=10,
        save_total_limit=1,  
        load_best_model_at_end=True,
        greater_is_better=False,
        seed = config['RANDOM_STATE'],
        output_dir = output_dir,
        report_to= "none", 
    )

    data_collator = DataCollatorForPrompt(dataset.tokenizer)

    peft_trainer = CustomSFTTrainer(
        model = peft_model,
        tokenizer = dataset.tokenizer,   # not pass tokenizer when using unsloth + SFTTrainer() <- AttributeError: 'NoneType' object has no attribute 'convert_ids_to_tokens'
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        args = peft_training_args,
        max_seq_length = max_length,
        dataset_num_proc = 2,
        packing = False, 
        random_state = config['RANDOM_STATE'],
        callbacks=[neptune_cb,early_stopping_cb],
    )
    peft_trainer.train()

    logger.info(dash_line)
    logger.info(f"**** Saving finetuned model to {output_dir}\n")
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    end_time = time.time()
    logger.info(f"Time cost: {end_time - start_time:.4f} seconds")

    
if __name__ == "__main__":
    main()


"""
Sample usage
python peft_trainer.py --config_file=configurations/peft-training-mtconan.yaml
"""