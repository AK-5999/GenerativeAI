# QLoRA: Theoretical Understanding
## 1. The Foundation: LoRA (Low-Rank Adaptation)
Basic Idea: Instead of updating all weights during fine-tuning, we add small "adapter" matrices.

```
Original Weight Matrix W (frozen):  [768 × 768] = 589,824 parameters

LoRA Decomposition:
W' = W + ΔW = W + (B × A)
where:
- B: [768 × r]  (r = rank, e.g., 8)
- A: [r × 768]
- ΔW = B × A: [768 × 768]

Trainable parameters: 768×8 + 8×768 = 12,288 (only 2% of original!)
```

## 2. The Innovation: Adding Quantization
QLoRA = Quantization + LoRA
```
Process Flow:
Step 1: Original Model (FP16/32)
         ↓
Step 2: Quantize to 4-bit (NF4)
         ↓
Step 3: Add LoRA adapters (FP16)
         ↓
Step 4: Train only LoRA parameters
```

## 3. How Quantization Works
4-bit NF4 (Normal Float 4) Quantization:
```
# Let's say we have 8 weights from a layer
original_weights = [-0.5, 0.2, 1.3, -0.8, 0.1, -0.3, 1.1, 0.7]
# Each stored as FP16 (16 bits) or FP32 (32 bits)Original weights: [-0.5, 0.2, 1.3, -0.8, ...]  # 16/32 bits each

Quantization process:
1. Find min/max & range values:
    min_weight = -0.8
    max_weight = 1.3
    range = max_weight - min_weight = 2.1
2. Create 16 bins (4-bit = 2^4 values)
    # Simplified uniform bins for illustration
    bin_width = range / 16 = 2.1 / 16 = 0.13125

    Bins:
    Bin 0:  [-0.8    to -0.669]  → Representative: -0.735
    Bin 1:  [-0.669  to -0.538]  → Representative: -0.603
    Bin 2:  [-0.538  to -0.406]  → Representative: -0.472
    Bin 3:  [-0.406  to -0.275]  → Representative: -0.341
    Bin 4:  [-0.275  to -0.144]  → Representative: -0.209
    Bin 5:  [-0.144  to -0.013]  → Representative: -0.078
    Bin 6:  [-0.013  to  0.119]  → Representative:  0.053
    Bin 7:  [ 0.119  to  0.250]  → Representative:  0.184
    Bin 8:  [ 0.250  to  0.381]  → Representative:  0.316
    Bin 9:  [ 0.381  to  0.513]  → Representative:  0.447
    Bin 10: [ 0.513  to  0.644]  → Representative:  0.578
    Bin 11: [ 0.644  to  0.775]  → Representative:  0.710
    Bin 12: [ 0.775  to  0.906]  → Representative:  0.841
    Bin 13: [ 0.906  to  1.038]  → Representative:  0.972
    Bin 14: [ 1.038  to  1.169]  → Representative:  1.103
    Bin 15: [ 1.169  to  1.300]  → Representative:  1.235
3. Map each weight to nearest bin
    Original → Bin Assignment:
    -0.5   → Falls in Bin 2  [-0.538 to -0.406]  → Index: 2
     0.2   → Falls in Bin 7  [0.119  to  0.250]  → Index: 7
     1.3   → Falls in Bin 15 [1.169  to  1.300]  → Index: 15
    -0.8   → Falls in Bin 0  [-0.8    to -0.669]  → Index: 0
     0.1   → Falls in Bin 6  [-0.013 to  0.119]  → Index: 6
    -0.3   → Falls in Bin 3  [-0.406 to -0.275]  → Index: 3
     1.1   → Falls in Bin 14 [1.038  to  1.169]  → Index: 14
     0.7   → Falls in Bin 11 [0.644  to  0.775]  → Index: 11
4. Store bin index (4 bits) + scaling factor + offset
    # Storage format
    quantized_indices = [2, 7, 15, 0, 6, 3, 14, 11]  # Each index uses only 4 bits
    scale_factor = 0.13125  # Store once for the entire tensor
    offset = -0.8  # Minimum value

# Memory usage:
# Original: 8 weights × 16 bits = 128 bits
# Quantized: 8 weights × 4 bits + 32 bits (scale) + 32 bits (offset) = 96 bits
# Savings: 25% (in practice, more with larger tensors)
Quantized: [bin_3, bin_7, bin_15, bin_2, ...] + scale_factor

5. DeQuantization:
    # To reconstruct (approximate) original values:
    def dequantize(index, scale_factor, offset):
        return offset + (index * scale_factor)

    # Reconstruction:
    Index 2  → -0.8 + (2 × 0.13125)  = -0.538  (original: -0.5)
    Index 7  → -0.8 + (7 × 0.13125)  = 0.119   (original: 0.2)
    Index 15 → -0.8 + (15 × 0.13125) = 1.169   (original: 1.3)
    # etc...

Overall Visual Representation:
Original (FP16):  [-0.5] [0.2] [1.3] [-0.8] [0.1] [-0.3] [1.1] [0.7]
                    16b   16b   16b    16b    16b    16b    16b   16b  = 128 bits

                              ↓ Quantization

Quantized (4-bit): [0010][0111][1111][0000][0110][0011][1110][1011] + scale + offset
                     4b    4b    4b    4b    4b    4b    4b    4b      32b     32b = 96 bits

Binary representation:
0010 = 2 (decimal)
0111 = 7 (decimal)
1111 = 15 (decimal)
etc...

Note: NF4 Specific Optimization:
        NF4 doesn't use uniform bins. Instead, it uses bins optimized for normal distribution:
        # NF4 bin values (pre-computed for optimal normal distribution coverage)
        nf4_values = [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0, 
              0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]

        # These values are chosen to minimize quantization error for normally distributed weights
```
## 4. The Training Process
- Forward Pass:
    1. Input → 
    2. Dequantize 4-bit weights to FP16 →
    3. Compute: Output = (W_quantized + B×A) × Input →
    4. Loss calculation
- Backward Pass:
    1. Compute gradients →
    2. Update ONLY LoRA weights (B, A) →
    3. Base model W stays frozen and quantized

## 5. Mathematical Representation
- For each layer:
    - y = (W_q + ΔW) × x = (Dequantize(W_4bit) + B×A) × x
    - where:
        - W_q: Quantized base weights (frozen)
        - B×A: LoRA adaptation (trainable)
        - x: Input
        - y: Output
## 6. Memory Savings Breakdown
- Standard Fine-tuning (Mistral 7B):
    - Model: 7B × 2 bytes (FP16) = 14GB
    - Gradients: 14GB
    - Optimizer states: 28GB (Adam)
    - Total: ~56GB

- QLoRA Fine-tuning:
    - Quantized Model: 7B × 0.5 bytes = 3.5GB
    - LoRA parameters: ~50M × 2 bytes = 0.1GB
    - Gradients (LoRA only): 0.1GB
    - Optimizer states (LoRA only): 0.2GB
    - Total: ~4GB
## 7. Why It Works
1. Information Preservation: 4-bit quantization preserves most important information
2. Low-Rank Hypothesis: Fine-tuning changes are low-rank in nature
3. Separate Precision: Adapters in high precision capture task-specific nuances
## 8. Visual Flow Diagram
```
Input Text
    ↓
[Embedding Layer]
    ↓
[Transformer Block 1]
    ├─ Attention
    │   ├─ Q: W_q^Q (4-bit) + B_Q×A_Q (FP16)
    │   ├─ K: W_q^K (4-bit) + B_K×A_K (FP16)
    │   └─ V: W_q^V (4-bit) + B_V×A_V (FP16)
    ↓
[Continue through layers...]
    ↓
Output (NER predictions)
```
## Key Takeaways:
- Base model stays frozen: Never update quantized weights
- LoRA captures adaptation: All task-specific learning in small matrices
- Mixed precision: Computation in FP16, storage in 4-bit
- Efficient backprop: Gradients only for <1% of parameters
---
# Fine-tuning Mistral 7B with QLoRA for Medical Entity Extraction

## Project Overview

This project aims to fine-tune the **Mistral 7B** model using **QLoRA** on a custom dataset of medical documents. 
The goal is to extract medical terminologies (such as diseases, medications, symptoms, etc.) and classify them correctly based on their entity type.

## Prerequisites:
- GPU: GeForce RTX 3050 GPU with 8GB of VRAM.
- Python: Python 3.10.0
- Libraries: transformers, datasets, torch, accelerate, peft, and qlora.

``` bash
pip install transformers datasets torch accelerate peft qlora
```

## 1. Dataset Preparation

I have a labeled dataset of 5,000 data points that contain medical terminologies. 
The dataset is annotated using formats like **BIO** (Begin, Inside, Outside).

### Example Dataset Format:
- **Text**: "Patient was diagnosed with type 2 diabetes mellitus and prescribed metformin."
- **Annotations (BIO format)**:
    - Patient O
    - was O
    - diagnosed O
    - with O
    - type B-Disease
    - 2 I-Disease
    - diabetes I-Disease
    - mellitus I-Disease
    - and O
    - prescribed O
    - metformin B-Medication
    - . O

#### Step 1: Convert the Data to the Right Format
I need to structure your dataset in a format that includes both the medical text and its respective annotations (labels for entities). I am using JSON format.
```
{
  "text": "The patient was diagnosed with type 2 diabetes and prescribed metformin.",
  "labels": [
    {"word": "The", "entity": "O"},
    {"word": "patient", "entity": "O"},
    {"word": "was", "entity": "O"},
    {"word": "diagnosed", "entity": "O"},
    {"word": "with", "entity": "O"},
    {"word": "type", "entity": "B-Disease"},
    {"word": "2", "entity": "I-Disease"},
    {"word": "diabetes", "entity": "I-Disease"},
    {"word": "and", "entity": "O"},
    {"word": "prescribed", "entity": "O"},
    {"word": "metformin", "entity": "B-Medication"},
    {"word": ".", "entity": "O"}
  ]
}
```
#### Step 2: Step 2: Load , convert into huggingface format and Tokenize the Data
I am using HuggingFace's datasets library to load the dataset and preprocess it for NER tasks. Tokenize the dataset so that each token has an associated label.
```python
from datasets import load_dataset
from transformers import MistralTokenizer

# Load the dataset (ensure you have your dataset in the correct format)
dataset = load_dataset('json', data_files='TrainingData.json')

# Split into train (80%), validation (10%), and test (10%)
train_dataset = dataset['train'].train_test_split(test_size=0.2)  # 80% train, 20% test
val_test_dataset = train_dataset['test'].train_test_split(test_size=0.5)  # Split the 20% into 50% validation and 50% test

# Now you have train, validation, and test datasets
train_dataset = train_dataset['train']
validation_dataset = val_test_dataset['train']
test_dataset = val_test_dataset['test']

# Initialize the tokenizer for Mistral 7B
tokenizer = MistralTokenizer.from_pretrained('mistralai/mistral-7b-instruct')

# Tokenization function that also aligns labels
def tokenize_and_align_labels(examples):
    # Tokenize the input texts
    tokenized_inputs = tokenizer(examples['text'], padding="max_length", truncation=True, is_split_into_words=True)
    labels = examples['labels']

    # Create the labels for NER task
    new_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to words
        label_ids = []

        # For each token, check the corresponding word and assign the label
        for word_id in word_ids:
            if word_id is None:
                # Padding token, assign -100
                label_ids.append(-100)
            else:
                # Get the entity label for the corresponding word_id (word token)
                label_ids.append(label[word_id]['entity'])

        new_labels.append(label_ids)
    
    # Add the new 'labels' to the tokenized input
    tokenized_inputs['labels'] = new_labels
    return tokenized_inputs

# Apply the tokenization and label alignment to the entire dataset
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# You can check if everything is correct by inspecting a sample
print(tokenized_datasets['train'][0])  # Print the first example of the train dataset

```
### Step 3. Training with QLoRA
I will fine-tune Mistral 7B using QLoRA to adapt the model with low-rank matrices, which helps reduce GPU memory consumption during training.
```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments, AutoModelForTokenClassification, BitsAndBytesConfig
import torch

# Configure quantization for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model with quantization (this makes it QLoRA)
model = AutoModelForTokenClassification.from_pretrained(
    'mistralai/Mistral-7B-Instruct-v0.1',
    quantization_config=bnb_config,  # This enables quantization
    device_map="auto",
    trust_remote_code=True,
    num_labels=len(label_list)  # Set your number of NER labels
)
```
#### 🔧 Importance of `prepare_model_for_kbit_training(model)`
This line is **essential when using quantized models (4-bit / 8-bit)** with LoRA or QLoRA fine-tuning.

##### ✅ What it does:
- Ensures **gradient flow** from inputs to LoRA layers (needed for learning).
- Converts sensitive layers (like LayerNorm) to **float32** for stability.
- Prepares the model for **adapter-based training** (e.g., LoRA).
- Makes the quantized model compatible with **PEFT (Parameter-Efficient Fine-Tuning)**.

##### 🤔 Why it's important:
- Without this line, **LoRA layers may not get updated** (no learning).
- Training may become **unstable or ineffective**.
- Essential step to make quantized models **trainable**.

##### 📌 When to use it:
- After loading a model in **4-bit / 8-bit precision** using `BitsAndBytesConfig`.
- Before applying **LoRA adapters** using `get_peft_model()`.
```python
# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA (same as before, but now applied to quantized model)
lora_config = LoraConfig(
    r=8,  # rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Mistral uses these names
    lora_dropout=0.1,
    bias="none",
    task_type="TOKEN_CLASSIFICATION"
)

# Apply LoRA to quantized model (making it QLoRA)
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# Training Arguments remain the same
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-4,  # Often needs to be slightly higher for QLoRA
    per_device_train_batch_size=4,  # May need to reduce due to quantization overhead
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    gradient_checkpointing=True,  # Helps save memory
    fp16=True,  # Mixed precision training
    save_strategy="epoch",
)

# Trainer setup
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

# Start training
trainer.train()
```
### Step 4: Parameter Tuning and Hyperparameter Optimization
Key Hyperparameters:
- Rank (r): The rank of the low-rank decomposition in QLoRA. Experiment with values like r=4, r=8, r=16, r=32.
   - Smaller values lead to faster training, but with potentially less capacity for adaptation.
   - Larger values provide more capacity but use more GPU memory.
- Lora Alpha (lora_alpha): A scaling factor that controls how much low-rank adaptation is applied.
   - Try values such as 8, 16, 32 for optimal adaptation.
- Learning Rate: Common starting point: 2e-5. Experiment with values like 1e-5 or 3e-5.
- Batch Size: Set between 8 and 16 based on GPU memory. Lower batch sizes may work better for memory-limited environments.
- Epochs: Start with 5 epochs to avoid overfitting, especially with small datasets like yours.

### Step 5: Experimenting:
- Tune parameters like r, lora_alpha, and learning rate. Monitor performance using validation data.
- Precision: The proportion of correctly identified entities out of all entities the model predicted.
- Recall: The proportion of correctly identified entities out of all actual entities in the dataset.
- F1-Score: The harmonic mean of precision and recall.
```python
from sklearn.metrics import precision_recall_fscore_support

# Get predictions on the test set
predictions = trainer.predict(tokenized_datasets['test'])

# Calculate precision, recall, and F1 score
y_true = predictions.label_ids
y_pred = predictions.predictions.argmax(axis=-1)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
```

### Step 6: Final Model Evaluation
Based on experimentation, the following configuration worked best:
- Rank (r): 8
- Lora Alpha (lora_alpha): 16
- Learning Rate: 2e-5
- Batch Size: 8
- Epochs: 5

### Why This Configuration?
- Rank (r=8): Balanced memory usage and model adaptation capability.
- Lora Alpha (lora_alpha=16): Strong enough adaptation without overfitting.
- Learning Rate (2e-5): Converged well on the validation set.
- Epochs (5): Enough to learn without overfitting.

The model achieved good performance with an F1-score of around 0.90 on the test set, which is suitable for practical use in medical document processing.
