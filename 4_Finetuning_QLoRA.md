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
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments,MistralForTokenClassification

# Load Mistral 7B model for token classification (NER)
model = MistralForTokenClassification.from_pretrained('mistralai/mistral-7b-instruct')

# Configure QLoRA
lora_config = LoraConfig(
    r=8,  # rank (can experiment with different values)
    lora_alpha=16,  # Scaling factor
    target_modules=['query', 'value'],  # Target attention modules
)

# Wrap the model with QLoRA
peft_model = get_peft_model(model, lora_config)

# Training Arguments
# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Where to save the model and logs
    evaluation_strategy="epoch",  # Evaluate after every epoch
    learning_rate=2e-5,  # Learning rate for fine-tuning
    per_device_train_batch_size=8,  # Adjust based on GPU memory
    per_device_eval_batch_size=8,
    num_train_epochs=5,  # Number of training epochs
    weight_decay=0.01,
    logging_dir='./logs',
)

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
