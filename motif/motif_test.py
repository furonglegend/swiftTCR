import numpy as np
from itertools import product


def build_markov_background(sequences, order=1):
    """
    Estimate Markov background model for amino acid sequences.
    """
    counts = {}
    for seq in sequences:
        for i in range(len(seq) - order):
            prefix = seq[i:i + order]
            next_char = seq[i + order]
            counts.setdefault(prefix, {})
            counts[prefix][next_char] = counts[prefix].get(next_char, 0) + 1

    probs = {}
    for prefix, cdict in counts.items():
        total = sum(cdict.values())
        probs[prefix] = {k: v / total for k, v in cdict.items()}

    return probs


def permute_sequence(seq, markov_model, order=1):
    """
    Generate one Markov-consistent permutation.
    """
    out = list(seq[:order])
    for i in range(order, len(seq)):
        prefix = ''.join(out[-order:])
        probs = markov_model.get(prefix)
        if probs is None:
            out.append(seq[i])
        else:
            chars, p = zip(*probs.items())
            out.append(np.random.choice(chars, p=p))
    return ''.join(out)


def motif_enrichment_score(seqs, motif):
    """
    Simple motif frequency score.
    """
    return sum(motif in s for s in seqs) / len(seqs)


def two_stage_motif_test(
    foreground,
    background,
    motif,
    n_perm=1000,
    order=1
):
    """
    Two-stage motif testing with permutation.
    """
    markov = build_markov_background(background, order)
    obs = motif_enrichment_score(foreground, motif)

    perm_scores = []
    for _ in range(n_perm):
        permuted = [permute_sequence(s, markov, order) for s in foreground]
        perm_scores.append(motif_enrichment_score(permuted, motif))

    p_value = (sum(s >= obs for s in perm_scores) + 1) / (n_perm + 1)
    return obs, p_value
