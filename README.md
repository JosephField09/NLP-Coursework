# DG3NLP Coursework: Emotion Classification of Tweets

## Setup
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Data
Dataset located in: `data/emotion_dataset.csv`

## Run
```bash
python emotion_classification.py
```

## Outputs
- Figures: `outputs/figures/`
- Models: `outputs/models/`

## Requirements
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- ~8GB RAM minimum

## Reproducibility
Fixed random seed (42) ensures identical results across runs.