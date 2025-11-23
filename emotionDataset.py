import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    """
    Custom PyTorch Dataset for BERT tokenization. 
    Handles conversion of text to BERT input format.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initialises the dataset
    
        Parameters:
            - texts: Array of text strings
            - labels: Array of integer labels (0-5)
            - tokenizer: BERT tokenizer
            - max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """Returns the number of samples"""
        return len(self.texts)
    
    def __getitem__(self, index):
        """
        Gets a single tokenized example.
    
        Parameters:
            - index: Index of the sample
    
        Returns:
            - Dictionary with input_ids, attention_mask, and labels
        """
        text = str(self.texts[index])
        label = self.labels[index]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=self.max_length,
            padding='max_length', 
            truncation=True,  
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }