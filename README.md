# Beyond Strategy: Stakeholder-Informed Counterspeech to Tackle Online Gender-Based Violence

This repository accompanies our paper:
> **"Beyond Strategy: Stakeholder-Informed Counterspeech to Tackle Online Gender-Based Violence"**

It contains all code, datasets, annotation/evaluation platform demos, and annotation guidelines used in the project. 

---

## 📁 Repository Structure

```bash
GBVCounterspeech
├── LICENSE
├── README.md
├── code/                           # Model training, generation, evaluation code
│   ├── configurations/            # YAML configs for training/evaluation
│   ├── dataset.py                 # Dataset loading & preprocessing
│   ├── dpo_trainer.py             # Direct Preference Optimisation trainer
│   ├── evaluation.py              # Evaluation pipeline
│   ├── generate.py                # Text generation logic
│   ├── peft_trainer.py            # Parameter-Efficient Fine-Tuning trainer
│   ├── preference.py              # Preference data processing
│   └── utils/                     # Helper modules
│       ├── __init__.py
│       ├── metrics.py             # Metric calculations
│       ├── processing.py          # Preprocessing helpers
│       ├── prompts.py             # Prompt templates
│       └── sampler.py             # Sampling strategies
├── data/                          # Dataset files
│   ├── gbv_cs.tsv                 # GBV Counterspeech dataset
│   └── gbv_cs_preference.json     # Binary preference dataset
├── guideline/                     # Annotation & evaluation guidelines
│   ├── GBV Annotation Guideline.pdf
│   ├── GBV Counterspeech Annotation Guideline.pdf
│   └── GBV Counterspeech Evaluation Guideline.pdf
└── platform/                      # Streamlit apps for annotation & evaluation
    ├── cs-app/                    # Counterspeech annotation interface
    ├── cs-eval-app/               # Evaluation interfaces (3 rounds)
    └── gbv-app/                   # GBV annotation interface
```

---

## 📊 Datasets

We release two datasets to support the development and evaluation of stakeholder-informed counterspeech against online gender-based violence (GBV):

- **GBV Counterspeech Dataset**  
  A dataset of GBV posts along with corresponding counterspeech responses. Each instance is annotated with GBV forms, GBV targets, and applied counterspeech strategies. We provide both majority labels and perspectives from multiple annotators.

- **GBV Counterspeech Preference Dataset**  
  A binary preference dataset constructed for Direct Preference Optimization (DPO) training. Each record includes a GBV post, two counterspeech responses, and binary feedback indicating which is preferred.

These datasets have been pushed to the Hugging Face Hub under our organisation account.  
**Note:** Due to anonymisation and review policies, links will be provided upon paper acceptance.


---

## 🚀 Getting Started

### Launch Annotation / Evaluation Platforms

Each app is built with Streamlit:

```bash
cd platform/<platform_folder>
streamlit run main.py
```

### Implement Experiments

Install dependencies, modify the config files in `code/configurations/` to suit your setup, then run codes.

```bash
cd code
pip install -r requirements.txt

python generate.py --config_file=configurations/generate-mtconan.yaml
python peft_trainer.py --config_file=configurations/peft-training-mtconan.yaml
python dpo_trainer.py --config_file=configurations/dpo-training-mtconan.yaml
python evaluation.py --config_file=configurations/evaluate.yaml
```

---

## 📄 Annotation Guidelines

All human annotation and evaluation tasks were guided by detailed instructions, available in the [`guideline/`](guideline/) folder.

These include:

- GBV context annotation
- Counterspeech strategy annotation
- Human evaluation of counterspeech effectiveness

---

## 📜 Citation

If you find this work useful, please cite our paper:

```
@inproceedings{anonymous2025beyond,
    title={Beyond Strategy: Stakeholder-Informed Counterspeech to Tackle Online Gender-Based Violence},
    author={Anonymous},
    booktitle={Under Review},
    year={2025}
}
```

---

## 🤝 Acknowledgements

We thank the stakeholders and community collaborators who contributed valuable insights during the annotation and evaluation processes.


