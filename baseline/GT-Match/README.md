# GT-Match Implementation

## Overview
This implementation of GT-Match [1] performs membership inference by:
1. Checking exact suffix matches between generations and ground truth
2. Labeling samples as members when exact match occurs

## Requirements
- Python 3.8+

## Usage
```bash
python compute_similarity.py
```

## Output
- Membership labels (1=member, 0=non-member)
- Accuracy score
- Confusion matrix

## Methodology
For each line i:
1. Compare generated_suffixes[i] with ground_truth[i]
2. Label as member (1) if exact string match
3. Label as non-member (0) otherwise

## References
[1] A. Al-Kaswan, M. Izadi, and A. Van Deursen, “Traces of memorisation
in large language models for code,” in Proceedings of the IEEE/ACM
46th International Conference on Software Engineering, 2024, pp. 1–12.