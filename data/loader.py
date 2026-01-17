"""
data/loader.py
Utilities to load repertoire/TCR datasets and split into task sets (T_pre, T_ret, T_val, T_test).

Assumptions:
 - Input files are CSV or TSV with at least columns for sequence and peptide (task id).
 - The file format and column names are configurable via config.
"""

from pathlib import Path
import pandas as pd
import random
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def read_table(path: Path, sep: str = None) -> pd.DataFrame:
    """
    Read a table in CSV/TSV. Auto-detects separator if not provided.
    """
    if sep is None:
        # quick heuristic: check file extension
        if path.suffix.lower() in [".tsv", ".txt"]:
            sep = "\t"
        else:
            sep = ","
    df = pd.read_csv(path, sep=sep, low_memory=False)
    return df


def collect_files(root: str, pattern: str = "*.csv") -> List[Path]:
    """
    Return list of files matching the glob pattern under root.
    """
    rootp = Path(root)
    return list(rootp.rglob(pattern))


def load_concat_files(root: str, pattern: str = "*.csv", sep: str = None) -> pd.DataFrame:
    """
    Load all matching files and concatenate into a single DataFrame.
    """
    files = collect_files(root, pattern)
    if not files:
        raise FileNotFoundError(f"No files found in {root} matching {pattern}")
    dfs = []
    for f in files:
        logger.info(f"Reading {f}")
        dfs.append(read_table(f, sep=sep))
    return pd.concat(dfs, ignore_index=True)


def filter_and_normalize(df: pd.DataFrame,
                         seq_col: str,
                         peptide_col: str,
                         label_col: str = None,
                         valid_aas: str = "ACDEFGHIKLMNPQRSTVWY",
                         min_len: int = 5,
                         max_len: int = 30) -> pd.DataFrame:
    """
    Apply basic filtering:
    - drop rows with missing sequence or peptide
    - uppercase sequences
    - remove sequences with invalid characters
    - drop sequences outside length bounds
    """
    df = df.copy()
    df = df.dropna(subset=[seq_col, peptide_col])
    df[seq_col] = df[seq_col].astype(str).str.upper().str.strip()
    valid_set = set(valid_aas)
    mask_valid = df[seq_col].apply(lambda s: set(s).issubset(valid_set))
    df = df[mask_valid]
    seq_len = df[seq_col].str.len()
    df = df[(seq_len >= min_len) & (seq_len <= max_len)]
    if label_col and label_col in df.columns:
        df[label_col] = df[label_col].astype(int)
    df = df.reset_index(drop=True)
    return df


def group_by_task(df: pd.DataFrame, peptide_col: str) -> Dict[str, pd.DataFrame]:
    """
    Group dataframe by peptide (task id). Returns mapping peptide -> DataFrame of rows for that peptide.
    """
    groups = {}
    for peptide, g in df.groupby(peptide_col):
        groups[str(peptide)] = g.reset_index(drop=True)
    return groups


def split_tasks(task_keys: List[str],
                seed: int = 42,
                pretrain_frac: float = 0.7,
                val_frac: float = 0.1,
                test_frac: float = 0.2) -> Tuple[List[str], List[str], List[str]]:
    """
    Deterministically split task keys into pretrain/val/test sets with given seed.
    Returns (pretrain_keys, val_keys, test_keys).
    """
    assert abs(pretrain_frac + val_frac + test_frac - 1.0) < 1e-6, "fractions must sum to 1"
    random.Random(seed).shuffle(task_keys)
    n = len(task_keys)
    n_pre = int(n * pretrain_frac)
    n_val = int(n * val_frac)
    pre = task_keys[:n_pre]
    val = task_keys[n_pre:n_pre + n_val]
    test = task_keys[n_pre + n_val:]
    return pre, val, test


def build_task_splits(df: pd.DataFrame,
                      seq_col: str,
                      peptide_col: str,
                      label_col: str = None,
                      config: dict = None) -> Dict[str, Dict]:
    """
    High-level convenience function:
    - filter & normalize
    - group by peptide
    - drop rare tasks
    - split into T_pre, T_val, T_test
    Returns dict with keys: all_tasks (mapping), pre_tasks (list), val_tasks (list), test_tasks (list)
    """
    if config is None:
        config = {}
    min_len = config.get("min_seq_len", 5)
    max_len = config.get("max_seq_len", 30)
    valid_aas = config.get("valid_aas", "ACDEFGHIKLMNPQRSTVWY")
    min_examples = config.get("min_examples_per_task", 5)
    seed = config.get("seed", 42)
    pre_frac = config.get("pretrain_task_frac", 0.7)
    val_frac = config.get("val_task_frac", 0.1)
    test_frac = config.get("test_task_frac", 0.2)

    df_f = filter_and_normalize(df, seq_col, peptide_col, label_col=label_col,
                                valid_aas=valid_aas, min_len=min_len, max_len=max_len)
    groups = group_by_task(df_f, peptide_col)
    # filter out rare tasks
    groups = {k: v for k, v in groups.items() if len(v) >= min_examples}
    task_keys = list(groups.keys())
    pre_keys, val_keys, test_keys = split_tasks(task_keys, seed=seed,
                                                pretrain_frac=pre_frac,
                                                val_frac=val_frac,
                                                test_frac=test_frac)
    return {
        "all_tasks": groups,
        "pre_keys": pre_keys,
        "val_keys": val_keys,
        "test_keys": test_keys
    }
