---
base_model: mistralai/Mistral-7B-Instruct-v0.1
library_name: peft
tags:
- base_model:adapter:mistralai/Mistral-7B-Instruct-v0.1
- lora
- transformers
- named-entity-recognition
- qlora
- token-classification
- instruction-tuning
---

# Model Card for Mistral-7B NER (LoRA Fine-tuned)

This model is a fine-tuned version of `mistralai/Mistral-7B-Instruct-v0.1` using **QLoRA** for **Named Entity Recognition (NER)** on clinical-style data including entities like diseases, medications, persons, and locations.

## Model Details

### Model Description

- **Base model:** mistralai/Mistral-7B-Instruct-v0.1
- **Fine-tuning method:** QLoRA (4-bit quantization with LoRA adapters)
- **Task:** Token classification for Named Entity Recognition (NER)
- **Entity types:** `B-Disease`, `I-Disease`, `B-Medication`, `I-Medication`, `B-Person`, `I-Person`, `B-Location`, `I-Location`, `O`
- **Precision:** fp16 (mixed precision training)
- **Adapter size:** LoRA rank 8 with target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`

- **Developed by:** [Your Name or Organization]
- **Model type:** Causal language model with LoRA adapters
- **Language(s):** English (clinical/medical text)
- **License:** [Specify, e.g., Apache 2.0 if applicable]
- **Finetuned from model:** mistralai/Mistral-7B-Instruct-v0.1

## Uses

### Direct Use

- Designed to extract medical entities from clinical-style sentences using token-level classification.
- Useful for information extraction, medical NLP tasks, and educational research on LoRA fine-tuning.

### Downstream Use

- Can be integrated into healthcare-focused NLP pipelines for structured data extraction, symptom detection, or medication tracking.

### Out-of-Scope Use

- Not intended for diagnostic or treatment recommendations.
- Not suitable for tasks requiring domain-specific medical accuracy beyond entity extraction.
- Avoid use in production healthcare applications without extensive validation.

## Bias, Risks, and Limitations

- May reflect biases present in the training data (e.g., overrepresentation of common diseases/medications).
- May misclassify rare or ambiguous entities.
- Medical domain requires high accuracy — incorrect labeling may have downstream consequences.

### Recommendations

- Review outputs manually in critical applications.
- Retrain or fine-tune further on domain-specific labeled datasets.
- Always validate predictions with clinical professionals when used in real-world scenarios.

## How to Get Started with the Model

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification,BitsAndBytesConfig
from peft import PeftModel
import torch
import numpy as np
# Step 1: Load base model and tokenizer
base_model = "mistralai/Mistral-7B-Instruct-v0.1"
adapter_model = "Aghori/mistral-medical-ner-qlora"
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    use_fast=True  # Optional, but usually better performance
)
# Fix: set pad_token
tokenizer.pad_token = tokenizer.eos_token
# Configure quantization for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # fallback to float16 for broader support
)

id2label = {
    0: "O",
    1: "B-Disease",
    2: "I-Disease",
    3: "B-Medication",
    4: "I-Medication",
    5: "B-Person",
    6: "I-Person",
    7: "B-Location",
    8: "I-Location"
}

# Load model with quantization (this makes it QLoRA)
model = AutoModelForTokenClassification.from_pretrained(
    base_model,
    quantization_config=bnb_config,  # This enables quantization
    device_map="auto",
    trust_remote_code=True,
    num_labels=len(id2label )  # Set your number of NER labels
)
model = PeftModel.from_pretrained(model, adapter_model)
# Make sure the model is on the correct device (CUDA if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

```

## Training Details

### Training Data

- Custom synthetic NER dataset consisting of ~50 labeled examples with clinical-style sentences including medications, diseases, names, and hospitals.

### Training Procedure

- **Quantization:** 4-bit QLoRA with NF4
- **LoRA config:** r=8, α=16, dropout=0.1
- **Precision:** fp16 mixed precision
- **Gradient checkpointing:** Enabled

#### Training Hyperparameters

- **Epochs:** 3–5
- **Batch size:** 4
- **Learning rate:** 2e-4
- **Loss function:** CrossEntropyLoss
- **Device:** Single GPU (e.g., RTX 3050)

## Evaluation

### Testing Data

- A manually labeled evaluation set with disease, medication, and person entities.

### Metrics

- **Training loss:** ~0.50
- **Eval loss:** ~0.05
- **F1 Score (weighted):** > 0.80 on test set
- **Evaluation speed:** ~9.176 samples/sec

### Summary

The model shows strong generalization after only 3–5 epochs of training on a small dataset, with a significant drop in evaluation loss, and high F1 score — indicating successful fine-tuning using QLoRA.

## Environmental Impact

- **Hardware Type:** NVIDIA RTX 3050 Laptop GPU
- **Training Time:** ~30 minutes
- **Compute Location:** Local
- **Cloud Provider:** N/A
- **Carbon Emitted:** Minimal due to short training duration and quantized model

## Technical Specifications

### Model Architecture

- Mistral-7B: Decoder-only transformer, fine-tuned for instruction-following.
- LoRA adapters applied to key/query/value/output projections for token classification.

### Software Stack

- `transformers`
- `peft`
- `accelerate`
- `bitsandbytes`
- `scikit-learn`

## Citation

**BibTeX:**
```bibtex
@misc{your2025mistralner,
  title={LoRA Fine-Tuned Mistral-7B for NER},
  author={Aman Kumar},
  year={2025},
  url={[https://huggingface.co/Aghori/mistral-medical-ner-qlora](https://huggingface.co/Aghori/mistral-medical-ner-qlora)}
}
```

## Model Card Contact

For questions, reach out to: [akmh20225@student.nitw.ac.in]

## Framework versions

- `transformers`: 4.33+
- `peft`: 0.7.1+
- `datasets`: 2.14+
- `accelerate`: 0.25+
- `bitsandbytes`: 0.41+
