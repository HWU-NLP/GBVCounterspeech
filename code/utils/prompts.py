from utils.processing import get_top_k_labels, get_i_th_label

### system prompts
system_prompt = """
You are an expert conversationalist who responds to the best of your ability. You generate counterspeech, a type of speech that actively challenges and pushes back against gender-based violence, aiming to reduce harm, eliminate hate speech, and promote respectful dialogue. You use up to 50 words.
"""

### chat templates
llama3_chat_template = ["""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>{SYSTEM}<|eot_id|>
<|start_header_id|>user<|end_header_id|>{INPUT}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>""",
"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>{SYSTEM}<|eot_id|>
<|start_header_id|>user<|end_header_id|>{INPUT}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>{OUTPUT}<|eot_id|>""",
"""{OUTPUT}<|eot_id|>"""]

phi4_chat_template = ["""<|im_start|>system<|im_sep|>{SYSTEM}<|im_end|>
<|im_start|>user<|im_sep|>{INPUT}<|im_end|>
<|im_start|>assistant<|im_sep|>""",
"""<|im_start|>system<|im_sep|>{SYSTEM}<|im_end|>
<|im_start|>user<|im_sep|>{INPUT}<|im_end|>
<|im_start|>assistant<|im_sep|>{OUTPUT}<|im_end|>"""]


### generation tasks
# zero-shot w/o strategy = baseline (MTConan, ...)
cs_generation_baseline_0 = (
    lambda gbv: f"""
Generate a single, coherent counterspeech response that effectively addresses the given gender-based violence text. 

#### Input Format:
- Gender-based violence text: {gbv}

#### Output:
""".strip()
)


# zero-shot w/ strategy (MTConan, ...)
cs_strategies = {
"Empathy and Affiliation": "focuses on promoting understanding, fostering peace and finding common ground with understanding tone.",
"Warning of Consequence": "cautions the speaker about the impact of their words via potential negative outcomes, such as legal, social, or personal consequences, with serious tone.",
"Hypocrisy or Contradiction": "discredits the argument through critical analysis by pointing out inconsistencies or double standards, with critical tone.",
"Shaming or Labelling": "attacks the speaker by using negative terms to emphasise the inappropriateness of their behaviour with accusatory or disapproving tone.",
"Denouncing": "condemns the hateful views, stating that they are wrong and unacceptable with firm tone.",
"Providing Facts": "corrects misinformation and false narratives, using facts and evidence to counter false claims with factual and corrective tone.",
"Humour or Sarcasm": "diminishes the comment by ridiculing it (or the perpetrator) via humouristic or sarcastic tone.",
"Questioning": "exposes flaws in the argument by probing its assumptions or intentions, often through asking clarifying questions with inquisitive and reflective tone.",
}

cs_generation_0_1 = (
    lambda gbv, cs_strategy: f"""
Generate a single, coherent counterspeech response that effectively addresses the given gender-based violence text by applying the specified strategies together in an integrated manner.

#### Input:  
- Gender-based violence text: {gbv}  
- Specified Strategies: 
1. Use strategy "{get_top_k_labels(cs_strategy, k=3)[0]}" that {cs_strategies[get_top_k_labels(cs_strategy, k=3)[0]]}

#### Output: 
""".strip()
)

cs_generation_0_2 = (
    lambda gbv, cs_strategy: f"""
Generate a single, coherent counterspeech response that effectively addresses the given gender-based violence (GBV) text by applying the specified strategies together in an integrated manner.

#### Input:  
- Gender-based violence text: {gbv}  
- Specified Strategies: 
1. Use strategy "{get_top_k_labels(cs_strategy, k=3)[0]}" that {cs_strategies[get_top_k_labels(cs_strategy, k=3)[0]]}
{f'2. Use strategy "{get_i_th_label(cs_strategy, i=1, k=3)}" that {cs_strategies[get_i_th_label(cs_strategy, i=1, k=3)]}' if get_i_th_label(cs_strategy, i=1, k=3)!='' else ''}

#### Output: 
""".strip()
)

gbv_forms = {
    "Dehumanisation": "reduces the target to a subhuman or non-human status",
    "Threatening": "expresses intent or encourages others to take action against the target",
    "Derogation": "uses explicit derogatory, insulting, or demeaning language, focusing on the target's character, abilities, or physical attributes",
    "Animosity": "expresses implicit or subtle hostility, often framed in a way that normalizes or downplays sexism via statements such as backhanded compliments",
    "Support of Hate Crimes": "glorifies, supports, or justifies hate crimes or discrimination",
}

cs_generation_form_0 = (
    lambda gbv, cs_strategy, gbv_form: f"""
Generate a single, coherent counterspeech response that effectively addresses the given gender-based violence text by applying both strategies together in an integrated manner, based on the form of the gender-based violence.
The gender-based violence text contains {get_top_k_labels(gbv_form, k=1)[0]} that {gbv_forms[get_top_k_labels(gbv_form, k=1)[0]]}.

#### Input:  
- Gender-based violence text: {gbv}  
- Specified Strategies: 
1. Use strategy "{get_top_k_labels(cs_strategy, k=3)[0]}" that {cs_strategies[get_top_k_labels(cs_strategy, k=3)[0]]}
{f'2. Use strategy "{get_i_th_label(cs_strategy, i=1, k=3)}" that {cs_strategies[get_i_th_label(cs_strategy, i=1, k=3)]}' if get_i_th_label(cs_strategy, i=1, k=3)!='' else ''}

#### Output: 
 """.strip()
)

cs_generation_form_target_0 = (
    lambda gbv, cs_strategy, gbv_form, gbv_target: f"""
Generate a single, coherent counterspeech response that effectively addresses the given gender-based violence text by applying both strategies together in an integrated manner, based on the form and the target of the gender-based violence. 
The gender-based violence text, directed to {get_top_k_labels(gbv_target, k=1)[0]}, contains {get_top_k_labels(gbv_form, k=2)[0]} that {gbv_forms[get_top_k_labels(gbv_form, k=2)[0]]}.

#### Input:  
- Gender-based violence text: {gbv}  
- Specified Strategies:
1. Use strategy "{get_top_k_labels(cs_strategy, k=3)[0]}" that {cs_strategies[get_top_k_labels(cs_strategy, k=3)[0]]}
{f'2. Use strategy "{get_i_th_label(cs_strategy, i=1, k=3)}" that {cs_strategies[get_i_th_label(cs_strategy, i=1, k=3)]}' if get_i_th_label(cs_strategy, i=1, k=3)!='' else ''}

#### Output:
""".strip()
)
