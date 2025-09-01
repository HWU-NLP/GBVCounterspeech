import os
import json
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from ast import literal_eval

from thefuzz import fuzz
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from math import exp

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from evaluate import load
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score

from utils.processing import save_json, stemming


class BaseMetric:
    '''
    Base class for all metrics.
    Each metric should inherit from this class and implement the compute method.
    '''
    def __init__(self, name: str):
        self.name = name

    def compute(
        self, 
        predictions: List[str], 
        references: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def validate_inputs(
        self, 
        predictions: List[str], 
        references: Optional[List[str]] = None
    ):
        if not predictions or not isinstance(predictions, list):
            raise ValueError("Predictions must be a non-empty list of strings.")
        if references is not None:
            if not isinstance(references, list) or len(predictions) != len(references):
                raise ValueError("References must be a list of the same length as predictions.")


# Specific metrics 
class BERTScore(BaseMetric):
    '''
    BERTScore metric for evaluating the similarity between model-generated and human-generated texts.
    '''
    def __init__(self, lang="en"):
        super().__init__("BERTScore")
        self.metric = load("bertscore")
        self.lang = lang

    def compute(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        
        self.validate_inputs(predictions, references)
        f1_scores, precisions = [], []
        for pred, ref in zip(predictions, references):
            result = self.metric.compute(predictions=[pred], references=[ref], lang=self.lang)
            f1_scores.extend(result["f1"])
            precisions.extend(result["precision"])
        return {
            "f1": float(np.average(f1_scores)),
            "precision": float(np.average(precisions))
        }
        
class BLEU(BaseMetric):
    '''
    BLEU metric for evaluating the similarity between model-generated and human-generated texts.
    '''
    def __init__(self):
        super().__init__("BLEU")
        self.metric = load("bleu")

    def compute(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        
        self.validate_inputs(predictions, references)
        results = self.metric.compute(predictions=predictions, references=references)
        return {
            "bleu": float(np.average(results["bleu"]))
        }

class SelfBLEU(BaseMetric):
    '''
    Self-BLEU metric for evaluating the diversity of model-generated texts.
    This metric computes the BLEU score of each generated text against all other generated texts.
    Regarding one sentence as hypothesis and the others as reference, we can calculate BLEU score for every generated sentence, and define the average BLEU score to be the Self-BLEU of the document.
    '''
    def __init__(self):
        super().__init__("SelfBLEU")
        self.metric = load("bleu")

    def compute(
        self, 
        predictions: List[str], 
        references: Optional[List[str]] = None
    ) -> Dict[str, float]:
        
        self.validate_inputs(predictions)
        results = []
        couples = []
        for i in range(len(predictions)):
            for j in range(1, len(predictions)):
                if i+j in range(len(predictions)) and (i,j) not in couples: 
                    score = self.metric.compute(predictions=[predictions[i]], references=[predictions[i+j]])['bleu']
                    results.append(score)
                    couples.append((j,i))
        return {
            "self_bleu": float(np.average(results)) if results else 0.0
        }

class TokenMatchAccuracy(BaseMetric):
    '''
    Token match metric for evaluating the morphological similarity between model-generated cs and prompt.
    
    fuzz.token_sort_ratio: compare token-sorted strings using Levenshtein similarity (not length-normalised)
    token_f1: compute precision, recall, and f1 based on token set overlap (report f1, length-normalised)
    jaccard_similarity: compute the proportion of overlapping tokens to total unique tokens (length-normalised)
    '''
    def __init__(self, prompts: List[str]):
        super().__init__("TokenMatchAccuracy")
        self.prompts = prompts
        self.stemmer = PorterStemmer()

    def symmetric_len_penalty(
        self,
        pred_len: float, 
        text_len: float, 
        alpha: float=0.5,
    ) -> float:
        ratio = pred_len / text_len if text_len > 0 else 1.0
        penalty = exp(-alpha * abs(np.log(ratio)))
        return penalty

    def fuzz_token_sort_ratio(
        self,
        predictions: List[str], 
        texts: List[str], 
    ) -> str:
        
        scores = []
        for pred, text in zip(predictions, texts):
            score = fuzz.token_sort_ratio(stemming(pred), stemming(text))
            penalty = self.symmetric_len_penalty(len(pred.split()), len(text.split()))  # add length penalty to reduce length bias
            adjusted_score = score * min(1.0, penalty)
            scores.append(adjusted_score)
            
        return float(np.average(scores))
    
    def token_f1(
        self,
        predictions: List[str], 
        texts: List[str], 
    ) -> List[str]:
        
        f1_scores = []
        precision_scores = []
        for pred, text in zip(predictions, texts):
            pred_tokens = set(self.stemmer.stem(w.lower()) for w in word_tokenize(pred))
            ref_tokens = set(self.stemmer.stem(w.lower()) for w in word_tokenize(text))

            tp = len(pred_tokens & ref_tokens)
            precision = tp / len(pred_tokens) if pred_tokens else 0
            recall = tp / len(ref_tokens) if ref_tokens else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
            f1_scores.append(f1)
            precision_scores.append(precision)
            
        return [float(np.average(f1_scores)), float(np.average(precision_scores))]

    def jaccard_similarity(
        self,
        predictions: List[str], 
        texts: List[str], 
    ) -> str:
        
        scores = []
        for pred, text in zip(predictions, texts):
            pred_tokens = set(self.stemmer.stem(w.lower()) for w in word_tokenize(pred))
            ref_tokens = set(self.stemmer.stem(w.lower()) for w in word_tokenize(text))

            intersection = pred_tokens & ref_tokens
            union = pred_tokens | ref_tokens
            score = len(intersection) / len(union) if union else 0
            scores.append(score)
            
        return float(np.average(scores))
    
    def compute(
        self, 
        predictions: List[str], 
        references: Optional[List[str]] = None
    ) -> Dict[str, float]:
        
        self.validate_inputs(predictions, self.prompts)

        # for target evaluation:
        processed_targets = []
        if '[' in self.prompts[0]:
            for e in self.prompts:
                c = literal_eval(e)
                c = list(set(c))
                processed_targets.append(' '.join(c))
            prompts = processed_targets
        else:
            prompts = self.prompts
        # print('token accuracy of text vs. ', prompts[:5])
        # print(predictions[:5])
        
        f1, precision = self.token_f1(predictions, prompts)
        return {
            "fuzz_token_sort_ratio": self.fuzz_token_sort_ratio(predictions, prompts),
            "token_f1": f1,
            "token_precision": precision,
            "jaccard_similarity": self.jaccard_similarity(predictions, prompts),
        }

class TokenMatchAccuracyPrompt(TokenMatchAccuracy):
    '''
    This is a specific implementation for the prompt.
    '''
    def __init__(self, prompts: List[str]):
        super().__init__(prompts)
        self.name = "TokenMatchAccuracy_prompt"

class TokenMatchAccuracyGBVTarget(TokenMatchAccuracy):
    '''
    This is a specific implementation for the gbv target.
    '''
    def __init__(self, prompts: List[str]):
        super().__init__(prompts)
        self.name = "TokenMatchAccuracy_gbv_target"

class TokenMatchAccuracyGBV(TokenMatchAccuracy):
    '''
    This is a specific implementation for the gbv text.
    '''
    def __init__(self, prompts: List[str]):
        super().__init__(prompts)
        self.name = "TokenMatchAccuracy_gbv"

class TokenMatchAccuracyInstruction(TokenMatchAccuracy):
    '''
    This is a specific implementation for pure instruction without gbv.
    '''
    def __init__(self, prompts: List[str]):
        super().__init__(prompts)
        self.name = "TokenMatchAccuracy_instruction"

class Classifier(BaseMetric):
    '''
    Classifier-based metric for evaluating model-generated texts.
    '''
    def __init__(self, model_path=str, model_name=str, labels=List, target_names=list):
        super().__init__("Classifier")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.labels = labels
        self.target_names = target_names

    def compute_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        df = pd.DataFrame.from_dict(classification_report(y_true, y_pred, output_dict=True, target_names=self.target_names))
        print('\nClassification report: ')
        print(classification_report(y_true, y_pred, target_names=self.target_names))
        return f1, precision, recall, accuracy, df

    def compute(
        self, 
        predictions: List[str], 
        ) -> Dict[str, float]:

        self.model.eval()
        classification = []
        for t in predictions:
            tokenized = self.tokenizer.encode_plus(t,return_tensors='pt')
            result = self.model(tokenized['input_ids'],tokenized['attention_mask'])

            # multilabel:
            if len(self.labels[0])>2:
                sigmoid = torch.nn.Sigmoid()
                logits = result.logits
                probs = sigmoid(logits.squeeze().cpu())
                pred = np.zeros(probs.shape)
                pred[np.where(probs >= 0.5)] = 1
            else:    
                # multiclass/binary:
                pred = torch.argmax(result['logits'].detach()).item()

            classification.append(pred)

        y_true = [np.array(x, dtype=float) for x in self.labels]
        f1, precision, recall, accuracy, df = self.compute_metrics(y_true, classification)
        return { 
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1_for_label": [{x: round(df._get_value('f1-score', x),4)} for x in self.target_names],
            }

class StrategyClassifier(Classifier):
    ''' 
    Strategy classifier for evaluating model-generated texts.
    '''
    def __init__(self, model_path: str, model_name: str, labels: List, target_names: list):
        super().__init__(model_path=model_path, model_name=model_name, labels=labels, target_names=target_names)
        self.name = "Classifier_strategy"

class GBVFormClassifier(Classifier):
    ''' 
    GBV form classifier for evaluating model-generated texts.
    '''
    def __init__(self, model_path: str, model_name: str, labels: List, target_names: list):
        super().__init__(model_path=model_path, model_name=model_name, labels=labels, target_names=target_names)
        self.name = "Classifier_gbv_form"
    

class MetricEvaluator:
    '''
    Class to evaluate multiple metrics on a set of predictions and references.
    It takes a list of metric instances, evaluates all, and returns a dictionary of results.
    '''
    def __init__(self, metrics: List[BaseMetric]):
        self.metrics = metrics

    def evaluate(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        
        results = {}
        for metric in self.metrics:
            try:
                # Check which arguments are required by the compute method
                if metric.name in ["BERTScore", "ROUGE", "BLEU"]:
                    result = metric.compute(predictions, references)
                elif metric.name == "SelfBLEU":
                    result = metric.compute(predictions)
                elif metric.name.startswith("TokenMatchAccuracy"):
                    result = metric.compute(predictions)
                elif metric.name.endswith("Classifier"):
                    result = metric.compute(predictions)
                else:
                    raise ValueError(f"Unsupported metric type: {metric.name}")

                results[metric.name] = result

            except Exception as e:
                results[metric.name] = {"error": str(e)}
        return results

    def save_all_results(
        self, 
        results: Dict[str, Dict[str, Any]], 
        save_path: str
    ):
        if not save_path.endswith(".json"):
            raise ValueError(f"{save_path} is not a JSON file.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as fp:
            json.dump(results, fp, indent=2)
            print(save_path)
            
