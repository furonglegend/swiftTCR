"""
data/preprocess.py
Tokenization and sequence preprocessing utilities for CDR3 / TCR sequences.

Provides:
 - amino-acid vocabulary & index mapping
 - functions to convert sequences to index arrays
 - padding utilities
 - optional simple k-mer featurizer
"""

from typing import Dict, List, Tuple, Sequence
import numpy as np

# Standard 20 amino acids (one-letter codes)
DEFAULT_AAS = "ACDEFGHIKLMNPQRSTVWY"

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(aas: str = DEFAULT_AAS, add_special: bool = True) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build mapping from amino-acid to integer index and reverse.
    Index 0 reserved for PAD if add_special True, 1 for UNK.
    Returns (stoi, itos).
    """
    stoi = {}
    itos = {}
    idx = 0
    if add_special:
        stoi[PAD_TOKEN] = idx
        itos[idx] = PAD_TOKEN
        idx += 1
        stoi[UNK_TOKEN] = idx
        itos[idx] = UNK_TOKEN
        idx += 1
    for aa in aas:
        stoi[aa] = idx
        itos[idx] = aa
        idx += 1
    return stoi, itos


def seq_to_indices(seq: str, stoi: Dict[str, int], max_len: int = None) -> List[int]:
    """
    Convert amino-acid sequence to list of indices using stoi.
    Unknown tokens map to UNK.
    If max_len provided, sequence is truncated to max_len.
    """
    unk_idx = stoi.get(UNK_TOKEN)
    if max_len is not None:
        seq = seq[:max_len]
    return [stoi.get(ch, unk_idx) for ch in seq]


def pad_sequences(batch_indices: List[List[int]], pad_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Right-pad sequences to max length in batch.
    Returns (padded_array [B, L], lengths [B]).
    """
    lengths = np.array([len(x) for x in batch_indices], dtype=np.int64)
    max_len = int(lengths.max()) if len(lengths) > 0 else 0
    B = len(batch_indices)
    padded = np.full((B, max_len), pad_idx, dtype=np.int64)
    for i, seq in enumerate(batch_indices):
        padded[i, :len(seq)] = seq
    return padded, lengths


def kmer_counts(seq: str, k: int = 3, vocab: str = DEFAULT_AAS) -> np.ndarray:
    """
    Simple k-mer count vector for sequence. Useful for lightweight featurization.
    Returns vector of length len(vocab)**k (can be large); prefer small k (2 or 3).
    """
    base = len(vocab)
    char2idx = {c: i for i, c in enumerate(vocab)}
    total = base ** k
    counts = np.zeros(total, dtype=np.float32)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        valid = all(ch in char2idx for ch in kmer)
        if not valid:
            continue
        idx = 0
        for ch in kmer:
            idx = idx * base + char2idx[ch]
        counts[idx] += 1.0
    # normalize by number of observed k-mers
    denom = max(1.0, sum(counts))
    return counts / denom
