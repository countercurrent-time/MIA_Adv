# Victim Models Fine-Tuning Defaults 

This README summarizes the **default training parameters** applied to all victim models.

---

## Global Default Parameters

Every victim model is trained with the following defaults:

- **Epochs**: 5
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW (weight_decay=1e-2, ε=1e-8)
- **Batch Size**: micro-batch = 2  
  (with 4-step gradient accumulation → effective batch size = 8)
- **Precision**: FP16
- **Hardware**: 2 × NVIDIA A6000 (48 GB each)

> - **Phi-2 (2.7B)** is **fully fine-tuned**.
> - **All 7B models** use **LoRA** with (r=16, α=32, dropout=0.1).



---

**Note:** Always validate on  benchmarks (HumanEval & APPS) after tuning. 

