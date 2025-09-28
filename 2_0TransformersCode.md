
```python
# Cell 1: Setup and Imports

import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```
```python
# Hyperparameters for a "small" transformer
VOCAB_SIZE = 1000       # Number of unique tokens
D_MODEL = 128           # Embedding dimension
NHEAD = 4               # Number of attention heads
NUM_ENCODER_LAYERS = 2  # Number of encoder layers
DROPOUT = 0.1
SEQ_LEN = 20            # Max sequence length
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
```

```python
# Cell 2: Utility Functions (Masking)

def generate_square_subsequent_mask(sz):
    """
    Generates an upper-triangular mask, preventing attention 
    to subsequent positions (causal masking).
    """
    # Create an upper-triangular matrix (triu) of ones, then transpose
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    
    # Fill the lower-triangular part with 0 (which becomes float(0.0) in the next step)
    # Fill the upper-triangular part with -inf (to mask out future tokens)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)
```

```python
# Cell 3: Positional Encoding Module

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate the division term for the sine/cosine arguments
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Shape: (max_len, 1, d_model) - for broadcasting during addition
        pe = pe.unsqueeze(0).transpose(0, 1) 
        
        # 'pe' is a buffer, not a parameter, so it won't be updated by gradients
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        # Add positional encoding to input embeddings (scaled input)
        x = x + self.pe[:x.size(0), :]
        return x
```

```python
# Cell 4: Small Transformer Model Definition

class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dropout):
        super(SmallTransformer, self).__init__()
        self.d_model = d_model
        
        # 1. Input Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder Block (using PyTorch's native modules)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout, 
            batch_first=False # Input shape is (seq_len, batch_size, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 4. Final Linear Layer (maps back to vocabulary size for prediction)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask):
        # src shape: (seq_len, batch_size)
        
        # Apply embedding and scaling
        src = self.embedding(src) * math.sqrt(self.d_model) 
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Pass through the Transformer Encoder
        output = self.transformer_encoder(src, src_mask) 
        
        # Final linear layer
        output = self.decoder(output)
        return output
```

```python
# Cell 5: Data Generation and Model Initialization

# --- Model Setup ---
model = SmallTransformer(
    vocab_size=VOCAB_SIZE, 
    d_model=D_MODEL, 
    nhead=NHEAD, 
    num_encoder_layers=NUM_ENCODER_LAYERS, 
    dropout=DROPOUT
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0) # Index 0 is often reserved for padding/ignored tokens
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Dummy Data Generation (Language Modeling Task) ---
# The task is to predict the next token in the sequence.
# 1. Source: Input sequence (tokens 1 to N-1)
src_data = torch.randint(1, VOCAB_SIZE, (SEQ_LEN, BATCH_SIZE), dtype=torch.long).to(device) 
# 2. Target: Shifted sequence (tokens 2 to N, plus a dummy token at the end)
tgt_data = torch.cat((src_data[1:], torch.zeros(1, BATCH_SIZE, dtype=torch.long).to(device)), dim=0)

# Generate the causal mask
src_mask = generate_square_subsequent_mask(SEQ_LEN)

print("Model initialized successfully.")
print(f"Source Data shape: {src_data.shape} (Seq_len, Batch_size)")
print(f"Target Data shape: {tgt_data.shape} (Seq_len, Batch_size)")
```

```python
# Cell 6: Training Loop

model.train()
start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    optimizer.zero_grad()

    # --- Forward Pass ---
    # output shape: (seq_len, batch_size, vocab_size)
    output = model(src_data, src_mask) 

    # --- Loss Calculation ---
    # Reshape for CrossEntropyLoss: (Batch_size * Seq_len, Vocab_size) vs (Batch_size * Seq_len)
    output_flat = output.reshape(-1, VOCAB_SIZE)
    tgt_flat = tgt_data.flatten()
    
    loss = criterion(output_flat, tgt_flat)

    # --- Backward Pass and Optimization ---
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Gradient clipping for stability
    optimizer.step()

    # --- Logging ---
    elapsed = time.time() - epoch_start_time
    print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | Time: {elapsed:.2f}s | Loss: {loss.item():.4f}")

end_time = time.time()
print("\n--- Training Complete ---")
print(f"Total training time for {NUM_EPOCHS} epochs: {end_time - start_time:.2f} seconds.")
```
