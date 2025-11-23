import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, f1_score, accuracy_score)
import torch
from torch.utils.data import Dataset
from transformers import (BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments)
from emotionDataset import EmotionDataset

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
print(torch.cuda.is_available())


# EMotion label mapping
EMOTIONS = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sadness', 5: 'surprise'}

# File paths
DATA_PATH = 'data/option1-training-dataset.csv'
MODELS_DIR = 'outputs/models'
FIGURES_DIR = 'outputs/figures'

# Model hyperparameters
TFIDF_MAX_FEATURES = 5000  # Maximum vocabulary size for TF-IDF
TFIDF_NGRAM_RANGE = (1, 2)  # Unigrams and bigrams
TEST_SIZE = 0.2  # 80-20 train-test split
BERT_MAX_LENGTH = 128  # Maximum token length for BERT
BERT_EPOCHS = 3  # Number of training epochs for BERT
BERT_BATCH_SIZE = 16  # Batch size for BERT training


def clean_text(text):
    # Cleans texts of links, mentions, hashtags, special characters and white space
    text = str(text).lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def plot_label_distribution(label_counts, save_path):
    plt.figure(figsize=(10, 6))
    colors = ["#ff1900", "#b700ff", "#ffcc00", "#ff82ac", "#049aff", "#00ff6a"]
    bars = plt.bar([EMOTIONS[i] for i in range(6)], 
                   [label_counts.get(i, 0) for i in range(6)], 
                   color=colors, 
                   edgecolor='black', 
                   linewidth=1)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.title('Emotion Distribution in Dataset', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Emotion', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Tweets', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(EMOTIONS.values()),
                yticklabels=list(EMOTIONS.values()),
                cbar_kws={'label': 'Count'},
                linewidths=0.5, 
                linecolor='gray')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_model_comparison(results_dict, save_path):
    models = list(results_dict.keys())
    accuracies = [results_dict[m]['accuracy'] for m in models]
    f1_scores = [results_dict[m]['f1'] for m in models]
    x = np.arange(len(models))
    width = 0.35
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, accuracies, width, label='Accuracy', 
                    color='#3498db', edgecolor='black', linewidth=1.2)
    bars2 = plt.bar(x + width/2, f1_scores, width, label='Weighted F1-score',
                    color='#e74c3c', edgecolor='black', linewidth=1.2)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(x, models, fontsize=10)
    plt.ylim(0, 1.1)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    # Print results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {model_name}")
    print('='*60)
    print(classification_report(y_true, y_pred, 
                                target_names=list(EMOTIONS.values()),
                                digits=4))
    print(f"Overall Accuracy:  {accuracy:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")
    return {'accuracy': accuracy, 'f1': f1}

def main():
    # Loading data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset from: {DATA_PATH}")
    print(f"\tShape: {df.shape}")
    print(f"\tColumns: {df.columns.tolist()}")

    # Cleaning data
    df['clean_text'] = df['text'].apply(clean_text)
    print(f"\nPreprocessing complete: {len(df)} valid samples")

    # Distributing labels
    label_counts = df['label'].value_counts().sort_index()
    print("\nLabel distribution:")
    for label in range(6):
        count = label_counts.get(label, 0)
        pct = (count / len(df)) * 100
        print(f"  {EMOTIONS[label]:8s}: {count:5d} ({pct:5.2f}%)")
    
    # Check for class imbalance
    max_count = label_counts.max()
    min_count = label_counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 3:
        print("\tWarning: Significant class imbalance detected")
    
    # Plot distribution
    plot_label_distribution(label_counts, f"{FIGURES_DIR}/label_distribution.png")

    # Train and Test
    X = df['clean_text'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED,
        stratify=y
    )
    print("\nTrainign complete:")
    print(f"\t- Training samples: {len(X_train)} ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"\t- Test samples:     {len(X_test)} ({TEST_SIZE*100:.0f}%)")
    
    # Model 1: TF-IDF and Logistic Regression
    print("")
    print("="*50)
    print("MODEL 1: TF-IDF + LOGISTIC REGRESSION")
    print(f"\t- Max features: {TFIDF_MAX_FEATURES}")
    print(f"\t- N-gram range: {TFIDF_NGRAM_RANGE}")
    
    # Create vectors
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=2,  # Ignore terms that appear in fewer than 2 documents
        max_df=0.95  # Ignore terms that appear in more than 95% of documents
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF vectors created:")
    print(f"\t- Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"\t- Matrix shape: {X_train_tfidf.shape}")

    # Train logistic regression
    model_lr = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        class_weight='balanced'
    )
    model_lr.fit(X_train_tfidf, y_train)

    # Predict and evaluate
    y_pred_lr = model_lr.predict(X_test_tfidf)
    lr_results = evaluate_model(y_test, y_pred_lr, "TF-IDF + Logistic Regression")

    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred_lr,
        'Confusion Matrix: TF-IDF + Logistic Regression',
        f'{FIGURES_DIR}/confusion_matrix_tfidf_lr.png'
    )
    
    #Save models
    with open(f'{MODELS_DIR}/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(f'{MODELS_DIR}/logistic_regression.pkl', 'wb') as f:
        pickle.dump(model_lr, f)
    
    # Model 2: Fine-tuned BERT
    print("")
    print("="*50)
    print("MODEL 2: FINE-TUNED BERT")
    # Load BERT 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=6, 
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )

    # Create datasets
    train_dataset = EmotionDataset(X_train, y_train, tokenizer, BERT_MAX_LENGTH)
    test_dataset = EmotionDataset(X_test, y_test, tokenizer, BERT_MAX_LENGTH)

    # Training config
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=BERT_EPOCHS,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        seed=RANDOM_SEED
    )

    # Create trainer
    trainer = Trainer(
        model=model_bert,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Training, predicting and evaluatinf
    trainer.train()
    predictions = trainer.predict(test_dataset)
    y_pred_bert = np.argmax(predictions.predictions, axis=1)
    bert_results = evaluate_model(y_test, y_pred_bert, "Fine-tuned BERT")

    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred_bert,
        'Confusion Matrix: Fine-tuned BERT',
        f'{FIGURES_DIR}/confusion_matrix_bert.png'
    )

    # Save Model
    trainer.save_model(f'{MODELS_DIR}/bert_finetuned')
    tokenizer.save_pretrained(f'{MODELS_DIR}/bert_finetuned')

    # Model comparison
    print("")
    print("="*50)
    print("COMPARING MODELS")
    results = {'TF-IDF + LR': lr_results, 'BERT': bert_results}

    # Comparison table
    print("\nPerformance Summary:")
    print("-" * 60)
    print(f"{'Model':<20} {'Accuracy':>12} {'F1-Score':>12}")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['accuracy']:>12.4f} {metrics['f1']:>12.4f}")
    print("-" * 60)

    # Work out the best model
    best_model = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"\nBest performing model: {best_model[0]}")
    print(f"\tF1-score: {best_model[1]['f1']:.4f}")

    # Plot comparison
    plot_model_comparison(results, f'{FIGURES_DIR}/model_comparison.png')
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise