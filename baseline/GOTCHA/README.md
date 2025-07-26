# GOTCHA

This repository of GOTCHA[1] implements a membership inference attack (MIA) pipeline on code-completion models using a surrogate-victim setup. It provides:

* Training a BERT-based classifier (`run.sh` + `run.py`) on features extracted from model outputs
* Inference of membership labels on new data (`pipeline.sh` + `mia.py`)



## Repository Structure

```
.
├── run.sh                    # wrapper to train classifiers over sample ratios
├── run.py                    # trains the BERT-based classifier
├── pipline.sh                # launches inference experiments over victim models and seeds
├── mia_3_component.sh        # called by pipline.sh
├── mia.py                    # loads trained classifier & runs MIA on victim outputs
```



## Prerequisites

Python >= 3.8
Install dependencies:
```bash
pip install torch transformers numpy scikit-learn tqdm
```

## Training the Classifier

### 1. Configure & Launch

Edit run.sh to set:

* `PER_NODE_GPU`
* `CUDA_VISIBLE_DEVICES`
* `MASTER_PORT`
* `SURROGATE_MODEL`
* `Percentage`
* Loop over `SAMPLE_RATIO` values

Then run:

```bash
bash run.sh
```

This calls run.py with arguments:

* `--lang`
* `--surrogate_model`
* `--sample_ratio`
* Optimization params (`batch_size`, `learning_rate`, `num_train_epochs`, …)
* Paths:

  * `--classifier_save_dir` (where to dump checkpoints)
  * `--prediction_data_folder_path` (your dataset folder)
  * `--lit_file` (path to `literals.json`)
  * `--classifier_model_path` (pretrained backbone for classification, e.g. CodeLlama)

### 2. run.py Flow

1. Parse args, set random seed
2. Load/prepare train+val JSON splits under `dataset/<lang>/<surrogate>/<ratio>/`
3. Instantiate a BERT‐based classifier (`AutoModelForSequenceClassification`) or tree‐aware variant
4. Train loop (with logging every `--evaluate_steps`)
5. Save best checkpoint under `--classifier_save_dir`



## Inference Pipeline

### 1. Configure & Launch

Edit pipeline.sh to set:

* `VICTIM_MODEL` array
* `SEED_GROUPS` mapping from CUDA device → seeds
* `MASTER_PORT`, `EPOCH`, etc.

Then run:

```bash
bash pipeline.sh
```

This will launch mia.py background jobs for each victim model, seed, and GPU.

### 2. mia.py Flow

1. Parse args (`--victim_model`, `--surrogate_model`, `--sample_ratio`, `--epoch`, `--seed`, etc.)
2. Load the BERT classifier checkpoint from `classifier_save/…`
3. Prepare test data splits via `dataset.*` utilities
4. Run `evaluate()` to compute:

   * Loss, Accuracy, Precision, Recall, F1
   * TPR, FPR, AUC
5. (Optional) `--save_results` dumps per‐sample predictions to JSON



## Key Scripts & Flags

### run.sh

* Controls training for sample ratios
* Invokes `python run.py --mode surrogate`

### run.py

* `--mode surrogate | <checkpoint-name>`
* Data prep, model init, training, checkpointing

### pipeline.sh

* Iterates victim models & seeds
* Exports env vars then calls `bash mia.py…`

### mia.py

* `--mode <victim>` picks which JSON test split
* `--epoch N` to load epoch‐N checkpoint
* `--save_results` to write out JSON with `prediction_label`



## References

* dataset/ contains utilities for:

  * `prepare_data()`, `divide_data()`
  * `ClassificationDataset`, collate functions
* model.py defines BERT‐based and Tree‐aware heads (`TBertT*`)

## References
[1] Z. Yang, Z. Zhao, C. Wang, J. Shi, D. Kim, D. Han, and D. Lo, “Gotcha! this model uses my code! evaluating membership leakage risks in code models,” IEEE Transactions on Software Engineering, 2024.