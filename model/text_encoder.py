import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class TextEncoder(nn.Module):
    """
    Text encoder using BERT to extract text features
    """
    def __init__(self, pretrained=True, hidden_dim=768):
        super(TextEncoder, self).__init__()
        
        # Load pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze BERT parameters
        if pretrained:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        # Projection layer (optional)
        self.projection = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU()
        )
        
        # Conditioning augmentation
        self.conditioning_augmentation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, text):
        """
        Extract text features from input description
        
        Args:
            text (str): Input text description
            
        Returns:
            torch.Tensor: Text embedding
        """
        # Tokenize the input text
        tokens = self.tokenizer(
            text, 
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        ).to(next(self.parameters()).device)
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert(**tokens)
            # Use [CLS] token embedding as sentence representation
            sentence_embedding = outputs.last_hidden_state[:, 0, :]
            # Get all token embeddings for word-level features
            word_embeddings = outputs.last_hidden_state
        
        # Project to desired dimension
        sentence_embedding = self.projection(sentence_embedding)
        
        # Apply conditioning augmentation for better sampling
        augmented_embedding = self.conditioning_augmentation(sentence_embedding)
        
        return {
            'sentence_embedding': augmented_embedding,
            'word_embeddings': word_embeddings
        }
    
    def encode_text(self, text):
        """Simplified interface that returns just the sentence embedding"""
        embeddings = self.forward(text)
        return embeddings['sentence_embedding']
