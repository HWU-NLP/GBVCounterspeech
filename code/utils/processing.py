import json
import re
import string
import ast
from loguru import logger
from collections import Counter
from typing import Any
from nltk.stem import PorterStemmer

def clean_text(text: str, min_len=4) -> str:
    punc = r"""()-[]{};:'"\<>/@#$%^&*_~"""

    if not isinstance(text, str):
        logger.debug(ValueError(f"encountered invalid format string: {text}"))
        return None

    text = re.sub(
        f"[^{re.escape(string.printable)}]",
        "",
        re.sub(
            r"[\?\.\!]+(?=[\?\.\!])",
            "",
            clean(
                text,
                fix_unicode=False,
                to_ascii=False,
                lower=False,
                no_line_breaks=True,
                no_urls=True,
                no_emails=True,
                no_punct=False,
                no_emoji=True,
                no_currency_symbols=True,
                lang="en",
                replace_with_url="",
                replace_with_email="",
            ),
        ),
    ).strip()

    for _ in punc:
        text = text.replace(_, "")
    text = text.replace(f". .", ". ").replace("  ", " ")
    text = text.strip()

    if len(text) < min_len:
        return "<EMPTY_TEXT>"
    else:
        return text.strip()


def load_json(fpath: str):
    if not fpath.endswith(".json"):
        raise ValueError(f"{fpath} not a json file")

    with open(fpath, "r") as fp:
        return json.load(fp)


def save_json(data: Any, fpath: str):
    if not fpath.endswith(".json"):
        raise ValueError(f"{fpath} not a json file")

    with open(fpath, "w") as fp:
        return json.dump(data, fp, indent=4)


def get_top_k_labels(input_data, k=3):
    if isinstance(input_data, str):
        input_list = ast.literal_eval(input_data) 
    else:
        input_list = input_data
    
    all_labels = [label.strip() for item in input_list for label in item.split(',')] 
    label_counts = Counter(all_labels)
    top_k_labels = [label for label, _ in label_counts.most_common(k)]
    
    return top_k_labels # return a list

def get_i_th_label(input_data, i=0, k=3):
    if len(get_top_k_labels(input_data, k=k)) > i:
        return get_top_k_labels(input_data, k=k)[i]
    else:
        return ""

ps = PorterStemmer()
def stemming(txt):
    txt_l = txt.lower().split(' ')
    new= []
    for w in txt_l:
        new.append(ps.stem(w))
    return ' '.join(new)

def clean_prompt(txt):
    txt = txt.split("user")[-1].strip().split("Input: ")[0].strip()
    if '#' in txt:
        txt = re.sub('#', '', txt)
    return txt.strip()