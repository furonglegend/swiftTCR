# SwiftRepertoire

**SwiftRepertoire** is a geometry-aware, parameter-efficient framework for few-shot T-cell receptor (TCR) repertoire analysis.  
The system synthesizes sparse task-specific adapters from prototype representations and couples rapid adaptation with interpretable motif discovery and calibrated statistical testing.

This repository contains the full research codebase accompanying the paper:

> *SwiftRepertoire: Sparse Geometry-Aware Adapter Synthesis for Few-Shot Immune Repertoire Analysis*  
> (submitted to ISME)

The implementation is designed for **reproducibility**, **modularity**, and **artifact evaluation**.

---

## 1. Overview

SwiftRepertoire addresses several practical challenges in TCR modeling:

- Scarce and long-tailed peptide annotations
- High-dimensional pretrained protein language model representations
- The need for fast task adaptation without full fine-tuning
- Interpretability requirements for biological and clinical deployment

The core idea is to:
1. Estimate task-specific adapters from few labeled samples.
2. Construct geometry-preserving prototypes from these adapters.
3. Retrieve sparse prototype combinations using compact task descriptors.
4. Apply calibrated motif testing on adapted representations.

---

## 2. Repository Structure

```text
configs/            # Experiment hyperparameters and paths
data/               # Dataset loading, preprocessing, and splits
models/             # Frozen encoder, probe head, adapter modules
adapters/           # Adapter estimation and canonicalization
analysis/            # SVD/PCA/Fisher analysis and diagnostics
prototypes/         # Prototype construction and coverage estimation
descriptors/        # Task descriptor computation
retrieval/          # Retrieval network and proximal solver
motif/              # Motif testing and statistical calibration
experiments/        # Training, evaluation, and ablation scripts
viz/                # Visualization utilities
utils/              # Metrics, logging, I/O, reproducibility helpers
tests/              # Unit tests for critical components
requirements.txt    # Python dependencies
