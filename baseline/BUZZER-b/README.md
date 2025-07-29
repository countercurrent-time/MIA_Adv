# BUZZER-b Implementation

## Overview
This implementation of BUZZER-b [1] performs membership inference attacks by:
1. Comparing original samples with lowercase-perturbed versions
2. Measuring representation change through cosine similarity
3. Ranking samples by sensitivity to casing perturbations
4. Predicting top 50% most sensitive samples as training data members

## Requirements
- Python 3.8+
- PyTorch
- Transformers library
- scikit-learn
- NumPy

## Usage
```bash
python compute_similarity.py
```

## Output
- Sensitivity scores for all samples (`sensitivity_true.txt`, `sensitivity_false.txt`)
- AUC and accuracy metrics
- ROC curve plot (`roc_curve.png`)

## Methodology
For each sample:
1. Extract CodeBERT embeddings of original and perturbed versions
2. Compute cosine similarity between representations
3. Calculate sensitivity as (1 - similarity)
4. Rank samples by sensitivity (higher = more likely member)

## References
[1] N. Carlini, F. Tr¨ amer, E. Wallace et al., “Extracting training data from large language models,” in 30th USENIX Security Symposium (USENIX Security 21), 2021, pp. 2633–2650.