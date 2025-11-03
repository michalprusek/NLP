"""Legal-BERT encoder for extracting embeddings from clause texts."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class LegalBERTEncoder(nn.Module):
    """Legal-BERT encoder with pooling for fixed-size embeddings."""

    def __init__(self, model_name: str = "nlpaueb/legal-bert-base-uncased", freeze_encoder: bool = False):
        """Initialize Legal-BERT encoder.

        Args:
            model_name: HuggingFace model identifier
            freeze_encoder: If True, freeze BERT weights (only train classifier)
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

        if freeze_encoder:
            print("Freezing Legal-BERT encoder weights")
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print("Legal-BERT encoder weights will be fine-tuned")

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """Encode texts to fixed-size embeddings.

        Args:
            texts: List of clause texts
            device: PyTorch device

        Returns:
            Tensor of shape (batch_size, hidden_size) with pooled embeddings
        """
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # Move to device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # Get BERT outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation (first token)
        # This is standard practice for classification tasks
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        return cls_embeddings

    def get_embedding_dim(self) -> int:
        """Get dimensionality of output embeddings."""
        return self.hidden_size
