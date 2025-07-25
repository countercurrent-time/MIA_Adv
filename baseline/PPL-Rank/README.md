# PPL-Rank Implementation

## Overview
This implementation of PPL-Rank [1] performs membership inference by:
1. Computing token-level perplexity for each sample
2. Ranking samples by perplexity (lowest first)
3. Predicting top 50% lowest-perplexity samples as members

## Requirements
- Python 3.8+
- PyTorch
- Transformers library
- scikit-learn
- NumPy

## Usage
```bash
python compute_perplexity.py
```

## Output
- Perplexity scores (`perplexity_true.txt`, `perplexity_false.txt`)
- AUC and accuracy metrics
- ROC curve plot (`roc_curve.png`)

## Methodology
For each sample:
1. Tokenize text using GPT-2 tokenizer
2. Compute language modeling loss
3. Calculate perplexity as exp(loss)
4. Rank samples by perplexity (lower = more likely member)

## References
[1] Z. Yang, Z. Zhao, C. Wang, J. Shi, D. Kim, D. Han, and D. Lo, “Un-
veiling memorization in code models,” in Proceedings of the IEEE/ACM
46th International Conference on Software Engineering, 2024, pp. 1–13.