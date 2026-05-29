# Positional Encoding in Transformers**

Transformers do not inherently capture token order since they process all tokens in parallel. Therefore, positional encodings are added to the input embeddings to provide the model with information about the position of tokens in the sequence.

### **Positional Encoding Types**

- **Sinusoidal Encoding**
  - **Description**: Uses sine and cosine functions of different frequencies to encode positions.
  - **Formula**:
    - \( PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}) \)
    - \( PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d}) \)
  - **Use Case**: Common in models like **GPT** and **BERT**.
  - **Pros**:
    - Simple and deterministic.
    - No need to learn additional parameters.
  - **Cons**:
    - Fixed encoding limits flexibility.
    - Might struggle with very long sequences.

- **Learned Positional Encoding**
  - **Description**: Each position has a corresponding learned vector during training.
  - **Formula**: No fixed formula; the position vector is learned during training.
  - **Use Case**: Widely used in **BERT** and other pre-trained transformer models.
  - **Pros**:
    - Flexible and can adapt to the dataset.
    - No limitations from fixed formulas.
  - **Cons**:
    - Increases model complexity.
    - Requires extra parameters and memory.

- **Absolute Positional Encoding**
  - **Description**: Encodes the absolute position of the tokens in the sequence.
  - **Formula**: Similar to sinusoidal encoding but directly encodes absolute positions.
  - **Use Case**: Used in classification and tasks where order is important (e.g., **BERT**).
  - **Pros**:
    - Direct encoding of position.
    - Works well for short sequences.
  - **Cons**:
    - Not optimal for long sequences.
    - Cannot capture relative position.

- **Relative Positional Encoding**
  - **Description**: Encodes the relative distance between tokens instead of their absolute positions.
  - **Formula**: Relative position information is added to the attention scores during self-attention.
  - **Use Case**: Used in models like **Transformer-XL**, **Reformer**, and **T5**.
  - **Pros**:
    - Handles long-range dependencies better.
    - Allows for flexible sequence lengths.
  - **Cons**:
    - More complex to implement.
    - May introduce bias if not handled well.

- **RoPE (Rotary Positional Encoding)**
  - **Description**: Rotates the query and key vectors to incorporate relative positions, making them more dynamic.
  - **Formula**: \( Q_{rot} = Q \cdot \cos(\theta) + K \cdot \sin(\theta) \), where \( \theta \) is a rotating factor.
  - **Use Case**: Applied in models like **RoBERTa**, **Swin Transformer**, and **GPT-NeoX**.
  - **Pros**:
    - Highly efficient for long sequences.
    - Great for handling very large contexts.
  - **Cons**:
    - Computationally expensive.
    - May introduce instability in certain settings.

- **ALiBi (Attention with Linear Biases)**
  - **Description**: Incorporates linear biases directly in the attention matrix to account for the distance between tokens.
  - **Formula**: Adds biases linearly to the attention scores, influenced by the relative distance.
  - **Use Case**: Used in **ALiBi Transformers**, useful for efficiently handling long sequences.
  - **Pros**:
    - Handles long-range dependencies without relying on learned embeddings.
    - Efficient and scalable.
  - **Cons**:
    - Might require careful tuning for optimal performance.
    - Not as widely adopted as other methods.

---

### **Summary of Positional Encodings**

- **Sinusoidal Encoding** is often used in **GPT** and **BERT** and is computationally simple.
- **Learned Positional Encoding** allows the model to learn a more flexible encoding but increases complexity.
- **Absolute Positional Encoding** is a deterministic and direct method, but may struggle with long sequences.
- **Relative Positional Encoding** excels in handling long-range dependencies and flexibility.
- **RoPE** enhances relative encoding by rotating vectors, which works well for very large sequence contexts.
- **ALiBi** avoids the need for traditional positional embeddings, offering efficiency for longer sequences.

---