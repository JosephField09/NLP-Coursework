import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from emotionDataset import EmotionDataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Checks if GPU is available for BERT (GPU much faster than CPU)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Create output folders if not already there
os.makedirs('outputs/models', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)

# EMotion label mapping
EMOTIONS = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sadness', 5: 'surprise'}

def clean_text(text):
    """
    Cleans the input text for:
        - Integer inputs
        - Removes links 
        - Removes mentions 
        - Removes hashtags
        - Keeps only letters and spaces
        - Removes extra whitespace
    
    Parameters:
        - text: The raw text string
    
    Returns:
        - text: The cleaned text string
    """
    text = str(text).lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def load_data(file_path):
    """
    Loads the data from the dataset at the input file_path
    
    Parameters:
        - file_path: Path location of .csv file
    
    Returns:
        - df: DataFrame with the loaded data
    """
    # Load the data
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    df = pd.read_csv(file_path)

    # Show some basic stats
    print(f"\tLoaded {len(df)} tweets")
    print(f"\tColumns: {df.columns.tolist()}")
    print(f"\tDataset shape: {df.shape}")
    print(f"\tFirst few items:")
    print(f"\t\t{df.head()}")
    return df

def plot_label_distribution(df, save_path):
    """
    Creates and saves a bar chart of emotion label distribution
    
    Parameters:
        - df: The DataFrame that is being used
        - filepath: Path to save the complete chart
    """
    # Distributing labels
    label_counts = df['label'].value_counts().sort_index()
    print("\nLabel distribution:")
    for label in range(6):
        count = label_counts.get(label, 0)
        pct = (count / len(df)) * 100
        print(f"  {EMOTIONS[label]}: {count} ({pct:1f}%)")

    # Create bar chart
    plt.figure(figsize=(10, 6))
    colors = ["#ff1900", "#b700ff", "#ffcc00", "#ff82ac", "#049aff", "#00ff6a"]
    plt.bar([EMOTIONS[i] for i in range(6)], 
            [label_counts.get(i, 0) for i in range(6)], 
            color=colors)
    plt.title('Emotion Distribution in Dataset', fontsize=14)
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Number of Tweets', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """
    Creates and saves a confusion matrix visualisation
    
    Parameters:
        - y_true: True labels
        - y_pred: Predicted labels
        - title: Plot title
        - save_path: Path to save complete matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(EMOTIONS.values()),
                yticklabels=list(EMOTIONS.values()))
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def evaluate_model(y_true, y_pred, model):
    """
    Prints evaluation metrics for a model
    
    Parameters:
        - y_true: True labels
        - y_pred: Predicted labels
        - model: Name of the model used
    
    Returns:
        - Dictionary with accuracy and f1 scores
    """
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS: {model}")
    print('='*50)
    # Print classification report
    print(classification_report(y_true, y_pred, 
                                target_names=list(EMOTIONS.values())))
    # Calculate accuracy and F1 score
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Overall Accuracy:  {accuracy:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")
    return {'accuracy': accuracy, 'f1': f1}

def train_tfidf_lr_model(X_train, X_test, y_train, y_test):
    """
    Trains TF-IDF + Logistic Regression baseline model
    
    Parameters:
        - X_train, X_test: Training and test text data
        - y_train, y_test: Training and test labels
    
    Returns:
        - Dictionary with results and predictions
    """
    print("")
    print("="*50)
    print("MODEL 1: TF-IDF + LOGISTIC REGRESSION")
    print("="*50)
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=5000,  
        ngram_range=(1, 2),  
        min_df=2 # Ignore very rare words
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")

    # Train logistic regression
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    model.fit(X_train_tfidf, y_train)
    print("Training complete")

    # Make predictions and evaluate
    y_pred = model.predict(X_test_tfidf)
    results = evaluate_model(y_test, y_pred, "TF-IDF and Logistic Regression")

    # Create confusion matrix
    plot_confusion_matrix(y_test, y_pred, 'Confusion Matrix: TF-IDF + Logistic Regression', 'outputs/figures/confusion_matrix_lr.png')

    # Save models with pickle
    with open('outputs/models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('outputs/models/logistic_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return{
        'results': results,
        'predictions': y_pred,
        'model': model,
        'vectorizer': vectorizer
    }

def train_bert_model(X_train, X_test, y_train, y_test):
    """
    Train fine-tuned BERT model
    
    Parameters:
        - X_train, X_test: Training and test text data
        - y_train, y_test: Training and test labels
    
    Returns:
        - Dictionary with results and predictions
    """
    print("")
    print("="*50)
    print("MODEL 2: FINE-TUNED BERT")
    print("="*50)
    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=6  # 6 emotion classes
    )

    # Create dataset
    train_dataset = EmotionDataset(X_train, y_train, tokenizer, max_length=128)
    test_dataset = EmotionDataset(X_test, y_test, tokenizer, max_length=128)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        logging_steps=100,
        seed=42,
        no_cuda=not torch.cuda.is_available()
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train the model
    print("\nTraining BERT...")
    trainer.train()
    print("Training complete")

    # Make predictions and evaluate
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    results = evaluate_model(y_test, y_pred, "Fine-tuned BERT")

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, 'Confusion Matrix: BERT', 'outputs/figures/confusion_matrix_bert.png')

    # Save model with pickle
    trainer.save_model('outputs/models/bert_finetuned')
    tokenizer.save_pretrained('outputs/models/bert_finetuned')
    
    return {
        'results': results,
        'predictions': y_pred,
        'trainer': trainer
    }

def compare_models(tfidf_results, bert_results, save_path):
    """
    Compare performance of both models and create visualization.
    
    Parameters:
        - tfidf_results: Dictionary of TF-IDF + LR model results
        - bert_results: Dictionary of BERT model results
        - save_path: Path to save comparison graph
    
    Returns:
        - Dictionary with each model's accuracy and F1 score
    """
    print("")
    print("="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    # Extract metrics
    tfidf_acc = tfidf_results['results']['accuracy']
    tfidf_f1 = tfidf_results['results']['f1']
    bert_acc = bert_results['results']['accuracy']
    bert_f1 = bert_results['results']['f1']

    # Print comparison
    print(f"\nTF-IDF + Logistic Regression:")
    print(f"\tAccuracy: {tfidf_acc:.4f}")
    print(f"\tF1-score: {tfidf_f1:.4f}")
    
    print(f"\nFine-tuned BERT:")
    print(f"\tAccuracy: {bert_acc:.4f}")
    print(f"\tF1-score: {bert_f1:.4f}")

    # Determine better model
    improvement = bert_f1 - tfidf_f1
    if improvement > 0:
        print(f"\nBERT performs better by {improvement:.4f} F1-score")
    else:
        print(f"\nTF-IDF performs better by {-improvement:.4f} F1-score")
    
    # Create comparison plot
    models = ['TF-IDF + LR', 'BERT']
    accuracies = [tfidf_acc, bert_acc]
    f1_scores = [tfidf_f1, bert_f1]
    
    x = np.arange(len(models))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, accuracies, width, label='Accuracy', color='#3498db')
    plt.bar(x + width/2, f1_scores, width, label='F1-score', color='#e74c3c')
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.xticks(x, models, fontsize=11)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return {
        'tfidf_acc': tfidf_acc,
        'tfidf_f1': tfidf_f1,
        'bert_acc': bert_acc,
        'bert_f1': bert_f1
    }

def main():
    """
    Main function to run the program
    """
    # Load and clean data
    df = load_data('data/option1-training-dataset.csv')
    df['clean_text'] = df['text'].apply(clean_text)
    print(f"\nPreprocessing complete: {len(df)} valid samples")
    
    # Plot label distribution
    plot_label_distribution(df, 'outputs/figures/label_distribution.png')

    # Split data
    X = df['clean_text'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train Model 1: TF-IDF + Logistic Regression
    tfidf_results = train_tfidf_lr_model(X_train, X_test, y_train, y_test)
    
    # Train Model 2: BERT
    bert_results = train_bert_model(X_train, X_test, y_train, y_test)

    # Model comparison
    comparison = compare_models(tfidf_results, bert_results, 'outputs/figures/model_comparison.png')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise