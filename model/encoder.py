import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class FrozenEncoder(nn.Module):
    """
    Wrapper for a frozen pretrained language model encoder.
    The encoder produces contextualized sequence embeddings
    and is kept fixed during all downstream adaptation.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.to(self.device)

    @torch.no_grad()
    def forward(self, sequences, pooling: str = "cls"):
        """
        Args:
            sequences: list[str], raw amino-acid sequences
            pooling: "cls" or "mean"

        Returns:
            Tensor of shape [batch_size, hidden_dim]
        """
        encoded = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        outputs = self.encoder(**encoded)
        hidden = outputs.last_hidden_state  # [B, L, D]

        if pooling == "cls":
            return hidden[:, 0, :]
        elif pooling == "mean":
            mask = encoded["attention_mask"].unsqueeze(-1)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
