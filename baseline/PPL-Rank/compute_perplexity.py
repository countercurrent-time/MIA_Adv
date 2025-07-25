import numpy as np
import json
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import torch
import time

# Initialize embedded model
MODEL_NAME = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def extract_embeddings(text):
    """Extract text embedding vector"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)
    return outputs.squeeze(0).cpu().numpy()


def compute_perplexity(text):
    """calculate perplexity"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return torch.exp(loss).item()  # perplexity = exp(loss)

def load_txt(file_path):
    """Load text file"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

# Data path
input_dir = "../dataset/"
true_file = input_dir + "train_CodeLlama-7b-hf_victim_infer.txt"
false_file = input_dir + "test_CodeLlama-7b-hf_victim_infer.txt"

# Load data
print("Loading data...")
true_data = load_txt(true_file)
false_data = load_txt(false_file)


print(f"Loaded {len(true_data)} true samples and {len(false_data)} false samples.")

# calculate perplexities
print("Calculating perplexities...")
start_time = time.time()

true_perplexities = []
for i, txt_line in enumerate(true_data):
    true_perplexities.append(compute_perplexity(txt_line))
    if i % 100 == 0:
        print(f"Processed {i}/{len(true_data)} true samples")

false_perplexities = []
for i, txt_line in enumerate(false_data):
    false_perplexities.append(compute_perplexity(txt_line))
    if i % 100 == 0:
        print(f"Processed {i}/{len(false_data)} false samples")

print(f"perplexity calculation completed in {time.time()-start_time:.2f} seconds")

# Create labels and uniform similarity list
labels = [1] * len(true_perplexities) + [0] * len(false_perplexities)
all_perplexities = true_perplexities + false_perplexities

# Sort by perplexity
print("Sorting by perplexity...")
sorted_indices = np.argsort(all_perplexities)  # Sort from small to large
sorted_perplexities = np.array(all_perplexities)[sorted_indices]
sorted_labels = np.array(labels)[sorted_indices]

# The first 50% of the forecast is true
n = len(sorted_labels)
half = n // 2
predictions = [1] * half + [0] * (n - half)

# Calculate evaluation index
accuracy = accuracy_score(sorted_labels, predictions)
roc_auc = roc_auc_score(sorted_labels, sorted_perplexities)

# Calculate ROC curve
fpr, tpr, _ = roc_curve(sorted_labels, -np.array(all_perplexities))  # use minus perplexities
roc_auc = auc(fpr, tpr)

print("\nEvaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {roc_auc:.4f}")

# Draw ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
print("ROC curve saved as 'roc_curve.png'")

# Save similarity results
with open('perplexity_true.txt', 'w') as f:
    for s in true_perplexities:
        f.write(f"{s}\n")

with open('perplexity_false.txt', 'w') as f:
    for s in false_perplexities:
        f.write(f"{s}\n")
        
print("perplexity results saved.")