"""
data/dataset.py
PyTorch Dataset and DataLoader helpers for TCR sequences and task-level episodes.

Provides:
 - SingleSequenceDataset: for simple per-sequence supervised tasks
 - RepertoireDataset: maps task_id -> sequences for retrieval / prototype building
 - TaskEpisodeDataset: yields support/query splits for meta/few-shot learning
 - collate functions for batched padding
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from torch.utils.data import Dataset
import torch
from data.preprocess import seq_to_indices, pad_sequences, build_vocab

# Build default vocab once
DEFAULT_STOI, DEFAULT_ITOS = build_vocab()


class SingleSequenceDataset(Dataset):
    """
    Standard dataset that returns (input_indices, label, meta) per row.
    Expects data_rows: list of dicts with keys: 'sequence', 'label' (optional), 'task_id' (optional)
    """
    def __init__(self, data_rows: List[Dict], stoi: Dict[str, int] = DEFAULT_STOI, max_len: int = 30):
        self.rows = data_rows
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        seq = row["sequence"]
        indices = seq_to_indices(seq, self.stoi, max_len=self.max_len)
        label = row.get("label", -1)
        task_id = row.get("task_id", None)
        return {"indices": torch.tensor(indices, dtype=torch.long),
                "length": len(indices),
                "label": torch.tensor(label, dtype=torch.long) if label is not None else -1,
                "task_id": task_id}


def collate_single(batch):
    """
    Collate function for SingleSequenceDataset. Pads sequences to batch max length.
    """
    indices = [b["indices"].tolist() for b in batch]
    padded, lengths = pad_sequences(indices, pad_idx=DEFAULT_STOI[PAD_TOKEN])
    labels = [int(b["label"]) if isinstance(b["label"], (int, np.integer)) else -1 for b in batch]
    task_ids = [b.get("task_id", None) for b in batch]
    return {
        "indices": torch.tensor(padded, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "task_ids": task_ids
    }


class RepertoireDataset:
    """
    Lightweight wrapper around the grouped tasks mapping (task_id -> pandas.DataFrame).
    Provides convenient accessors for building prototypes, estimating adapters, etc.
    """
    def __init__(self, groups: Dict[str, 'pd.DataFrame'], seq_col: str = "cdr3", label_col: Optional[str] = None):
        self.groups = groups
        self.seq_col = seq_col
        self.label_col = label_col
        # Build an internal list of tasks for deterministic ordering
        self.task_ids = sorted(list(groups.keys()))

    def get_task_ids(self) -> List[str]:
        return self.task_ids

    def get_sequences_for_task(self, task_id: str) -> List[str]:
        df = self.groups[task_id]
        return df[self.seq_col].astype(str).tolist()

    def get_labeled_pairs_for_task(self, task_id: str) -> List[Tuple[str, int]]:
        if self.label_col is None:
            raise ValueError("No label_col was provided to RepertoireDataset")
        df = self.groups[task_id]
        return list(zip(df[self.seq_col].astype(str).tolist(), df[self.label_col].astype(int).tolist()))


class TaskEpisodeDataset(Dataset):
    """
    Dataset that yields episodes for meta/few-shot training.
    Each item is a dict containing:
      - task_id
      - support: list of (sequence, label)
      - query: list of (sequence, label)
    Parameters:
      - groups: mapping task_id -> pandas.DataFrame
      - n_support: number of support examples per episode
      - n_query: number of query examples per episode
      - episodes_per_epoch: how many episodes to sample per epoch
    """
    def __init__(self, groups: Dict[str, 'pd.DataFrame'],
                 seq_col: str = "cdr3",
                 label_col: str = "label",
                 n_support: int = 5,
                 n_query: int = 15,
                 episodes_per_epoch: int = 100,
                 seed: int = 42):
        self.groups = groups
        self.seq_col = seq_col
        self.label_col = label_col
        self.n_support = n_support
        self.n_query = n_query
        self.episodes_per_epoch = episodes_per_epoch
        self.task_ids = sorted(list(groups.keys()))
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.episodes_per_epoch

    def __getitem__(self, idx):
        # Sample a random task
        task_id = self.rng.choice(self.task_ids)
        df = self.groups[task_id]
        n_total = len(df)
        # sample without replacement
        indices = self.rng.choice(n_total, size=min(n_total, self.n_support + self.n_query), replace=False)
        if len(indices) < self.n_support + self.n_query:
            # if insufficient examples, allow sampling with replacement
            indices = self.rng.choice(n_total, size=self.n_support + self.n_query, replace=True)
        support_idx = indices[:self.n_support]
        query_idx = indices[self.n_support:self.n_support + self.n_query]
        support = [(df.iloc[i][self.seq_col], int(df.iloc[i][self.label_col])) for i in support_idx]
        query = [(df.iloc[i][self.seq_col], int(df.iloc[i][self.label_col])) for i in query_idx]
        return {
            "task_id": task_id,
            "support": support,
            "query": query
        }


def batchify_episode(episode, stoi: dict = DEFAULT_STOI, max_len: int = 30):
    """
    Convert a raw episode to padded tensors for model consumption.
    Returns dict with support_indices, support_lengths, support_labels, query_indices, query_lengths, query_labels.
    """
    support_seqs, support_labels = zip(*episode["support"])
    query_seqs, query_labels = zip(*episode["query"])

    support_idx = [seq_to_indices(s, stoi, max_len=max_len) for s in support_seqs]
    query_idx = [seq_to_indices(s, stoi, max_len=max_len) for s in query_seqs]

    s_padded, s_lengths = pad_sequences(support_idx, pad_idx=DEFAULT_STOI[PAD_TOKEN])
    q_padded, q_lengths = pad_sequences(query_idx, pad_idx=DEFAULT_STOI[PAD_TOKEN])

    return {
        "task_id": episode["task_id"],
        "support_indices": torch.tensor(s_padded, dtype=torch.long),
        "support_lengths": torch.tensor(s_lengths, dtype=torch.long),
        "support_labels": torch.tensor(support_labels, dtype=torch.long),
        "query_indices": torch.tensor(q_padded, dtype=torch.long),
        "query_lengths": torch.tensor(q_lengths, dtype=torch.long),
        "query_labels": torch.tensor(query_labels, dtype=torch.long)
    }
