import json
import argparse
import yaml
import os
from os import walk
import pandas as pd
import time
from loguru import logger

from utils.metrics import (
    BERTScore, 
    BLEU, 
    SelfBLEU, 
    TokenMatchAccuracyPrompt, 
    TokenMatchAccuracyGBVTarget,
    TokenMatchAccuracyGBV,
    TokenMatchAccuracyInstruction,
    StrategyClassifier,  
    GBVFormClassifier,
)
from utils.metrics import MetricEvaluator
from utils.processing import save_json, clean_text, clean_prompt
from dataset import GBV_MTConan

def load_generated_data(json_path):
    print('\nLoading generated cs...')
    with open(json_path, "r") as f:
        data = json.load(f)
    # sort needed for matching original labels for evaluaton with classifiers
    data = sorted(data, key=lambda x: x['pair_id'])
    print([item["pair_id"] for item in data][:10])

    predictions = [clean_text(item["cs_output"]) for item in data]
    references = [item["cs_human"].strip() for item in data] 
    prompts = [item["prompt"].split("system")[-1].strip() for item in data] # w/ gbv
    gbv = [item["gbv"].strip() for item in data]
    instruction = [clean_prompt(item["prompt"]) for item in data] #w/o gbv

    return predictions, references, prompts, gbv, instruction

def load_labels(dataset_path=str, label=str):
    print('\nLoading true labels...')
    dataset = GBV_MTConan(
        file=dataset_path, 
        tokenizer='',
        max_length='',
        task= 'classify_bert',
        label=label,
        labels_map={},
        task_prompt='',
        text_columns='',
        batch_size=0,
        unsloth=False,
        instruct = '',
        chat_template = '',
        device=''
    )

    data = dataset.__data__().sort_values(by=['pair_id'])
    if label[0] == 'gbv_target':
        labels = data[label[0]].tolist()
    else:
        labels = data['label_encoded'].tolist()
    print('labels type: ', labels[:10])
    print('pair_id', data['pair_id'].tolist()[:10])
    target_labels = dataset.labels
    print('target_labels: ', target_labels)
    return labels, target_labels

def evaluate_cs(fpath, save_path, model_path=list, model_name=str, labels=list, target_names=list):
    # load data
    predictions, references, prompts, gbv, instruction = load_generated_data(fpath)
    # define metrics and evaluator
    metrics = [
        BERTScore(lang="en"),
        BLEU(),
        SelfBLEU(),
        TokenMatchAccuracyPrompt(prompts=prompts),
        TokenMatchAccuracyGBVTarget(prompts=labels[2]), 
        TokenMatchAccuracyGBV(prompts=gbv), 
        TokenMatchAccuracyInstruction(prompts=instruction), 
        StrategyClassifier(model_path=model_path[0], model_name=model_name, labels=labels[0], target_names=target_names[0]),
        GBVFormClassifier(model_path=model_path[1], model_name=model_name, labels=labels[1], target_names=target_names[1]),
    ]
    evaluator = MetricEvaluator(metrics)
    results = evaluator.evaluate(predictions, references)

    # save results
    evaluator.save_all_results(results, save_path)
    for metric, res in results.items():
        logger.info(f"\n{metric} Results:")
        for k, v in res.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate counterspeech given a generated file.")
    parser.add_argument("--config_file", required=True, help="Path to the config file")
    args = parser.parse_args()
    
    start_time = time.time()
    # Check if the config file exists
    assert os.path.exists(args.config_file), f"Config file '{args.config_file}' does not exist."
    
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)
        dpath = config['input_file']

        if config['multi_file']:
            file_list = []
            for (dirpath, _, filenames) in walk(dpath):
                for filename in filenames:
                    fpath = os.path.join(dirpath, filename)
                    if not fpath.endswith('.json'):
                        continue
                    file_list.append(fpath)
        else:
            file_list = [dpath]

    
    labels_strategy, target_labels_strategy = load_labels(config['dataset_file'], config['strategy_label'])
    labels_form, target_labels_form = load_labels(config['dataset_file'], config['form_label'])
    labels_target, _ = load_labels(config['dataset_file'], config['target_label'])
    
    labels = [labels_strategy, labels_form, labels_target]
    target_labels = [target_labels_strategy, target_labels_form]
    model_paths = [config['strategy_model_path'], config['form_model_path']]
    
    logger.info(f"files to evaluate: {file_list}")
    merged_results = {}
    for fpath in file_list: 
        start_time1 = time.time()
        logger.info(f"\n\n#### Evaluating file: {fpath}")
        
        save_path = os.path.join(config['output_file'], f"metrics_{fpath.split('/')[-1]}") 
        result = evaluate_cs(fpath, save_path, model_paths, config['model_name'], labels, target_labels)
        merged_results[fpath.split('/')[-1]] = result
        
        end_time1 = time.time()
        logger.info(f"Time cost: {end_time1 - start_time1:.4f} seconds for {fpath.split('/')[-1]}")
        
    # Save merged results
    merged_save_path = os.path.join(config['output_file'], "merged_results.json")
    save_json(merged_results, merged_save_path) 
    logger.info(f"\nAll merged results from diverse tasks and models saved to {merged_save_path}")

    end_time = time.time()
    logger.info(f"Time cost: {end_time - start_time:.4f} seconds")
    
if __name__ == "__main__":
    main()

"""
Sample usage
python evaluation.py --config_file=configurations/evaluate.yaml
"""