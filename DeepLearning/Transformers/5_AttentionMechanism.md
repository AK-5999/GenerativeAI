# Attention Mechanism
---

# Self Attention: A method to convert static embeddings into contextual aware embeddings.
## 1. Intro & The Core Concept of Self-Attention

In traditional Natural Language Processing (NLP) models (like standard Recurrent Neural Networks), processing text sequence-by-sequence often leads to a loss of context, especially over long distances. **Self-Attention** is the foundational mechanism that solves this by allowing a model to look at other words in the input sequence to get a better encoding for the current word.

* **Definition:** A mechanism that enables a model to weigh the importance of different words in a sequence relative to each other.
* **The Goal:** To dynamically capture "long-range dependencies" and contextual relationships in data without relying on sequential processing.

---

## 2. The Problem of "Average Meaning"

Before diving into how Self-Attention works, it is crucial to understand the limitation it directly addresses in older vector embedding techniques (like Word2Vec or GloVe):

* **Static Embeddings:** In older architectures, a word like *"bank"* has a single, fixed vector representation.
* **The Issue:** If you say *"I am going to the bank to deposit money"* versus *"I am sitting on the river bank,"* a static model provides the exact same mathematical vector for *"bank"* in both sentences. This results in an **"average meaning"** that fails to capture polysemy (words with multiple meanings).
* **The Self-Attention Fix:** Instead of assigning a fixed meaning, Self-Attention look at the surrounding context words (like *"deposit/money"* vs. *"river"*) and **dynamically recalculates** the word vector based on its environment.

---
![alt text](images/image-4.png)

Note: 
* Speed is one advantage of this method as all key,querry,value can be calculated parallely and attention score also can calcualted parallely.
* No tuning parameter is available.
* It is helpful in General contextual understanding. it may fail on task specific like translation of proverbs into hindi.
---

## 3. How Self-Attention Works (The Mechanism)

The video breaks down how a sentence is mathematically transformed to inject context. While the underlying math relies on Linear Algebra, the process follows these conceptual steps:

### Step 1: Input Embeddings

Each word in the input sequence is converted into a continuous vector (an embedding).

* Let's say the input is a sequence of vectors: $X = [x_1, x_2, ..., x_n]$.

### Step 2: Creating Queries, Keys, and Values

For every input vector $x_i$, the model projects it into three distinct spaces by multiplying it with three trained weight matrices ($W_Q, W_K, W_V$):

* **Query ($Q$):** "What am I looking for?" (The current word asking about its relationship to others).
* **Key ($K$):** "What do I contain?" (The label or index of every word in the sentence).
* **Value ($V$):** "What is my actual content?" (The actual information the word carries).

### Step 3: Calculating Attention Scores

To see how much focus word $A$ should place on word $B$, the model takes the Dot Product of the Query of word $A$ ($q_A$) and the Key of word $B$ ($k_B$).


![alt text](images/image-0.png)

### Step 4: Scaling and Softmax

1. **Scaling:** The scores are divided by the square root of the dimension of the key vectors ($\sqrt{d_k}$). This prevents the dot products from growing excessively large, which can cause gradient issues during training.
* dimensionality of all key, querry and value will be same equal to $$ d_k $$.
* High dimensional vector's dot product have high variance. High vairance means some numbers is very large cmpared to others. so, if we apply softmax over such numbers, bydefualt it will map high numbers to high probablities and low numbers to low probabilities. Hence, during BPT, gradient will be vanish for low probablities number and training process will focus only large numbers.
* Variation of product of any two matrix with same dimension 'd' is equals to d. If the variance is $d_k$, it means the standard deviation is $\sqrt{d_k}$.By dividing the dot product by its standard deviation ($\sqrt{d_k}$), we mathematically force the variance of the result back down to 1. 
2. **Softmax:** A Softmax function is applied to turn these scores into probabilities (values between 0 and 1 that sum up to 1). This tells us exactly what percentage of attention a word should pay to every other word.

### Step 5: Weighted Sum (The Final Output)

Finally, the attention probabilities are multiplied by the corresponding **Value ($V$)** vectors. Summing these up gives the final, context-aware vector representation for that specific word.

![alt text](images/image-1.png)

---

## Summary of Benefits

* **Contextual Awareness:** Words change their mathematical meaning depending on the sentence they are in.
* **Parallelization:** Unlike RNNs which process word-by-word, Self-Attention allows the entire sentence to be processed simultaneously, making training significantly faster on modern hardware (GPUs).
* **Long-Range Connections:** A word at the very beginning of a long paragraph can easily "attend" to a word at the very end without information leaking or degrading over time.


---

# Multi head attention

![alt text](images/image-2.png)

## Core Concept: Why Multi-Head Attention?

While standard **Self-Attention** allows a model to look at other words in a sentence to understand the context of a specific word, it has a major limitation: it only focuses on **one relationship or perspective at a time** (a single attention distribution).

**Multi-Head Attention (MHA)** solves this by splitting the attention mechanism into multiple "heads" running in parallel. This allows the model to jointly attend to information from different representation subspaces at different positions.

---

## Key Comparisons: Self-Attention vs. Multi-Head Attention

| Feature | Single-Head Self-Attention | Multi-Head Attention |
| --- | --- | --- |
| **Perspectives** | Captures only one type of relationship at a time. | Captures multiple relationships simultaneously (e.g., syntactic, semantic, long-range). |
| **Output** | Generates a single attention matrix. | Generates multiple attention matrices that are combined. |
| **Information** | Risk of the model focusing too much on its own position. | Encourages the model to focus on diverse parts of the sequence. |

---

## How It Works (Step-by-Step)

### 1. Linear Projection

Instead of performing a single attention function with the full dimensionality of the queries, keys, and values ($d_{model}$), the input embeddings are linearly projected $h$ times (where $h$ is the number of heads).

* **Queries ($Q$)**, **Keys ($K$)**, and **Values ($V$)** are split into lower-dimensional spaces:

$$d_k = d_v = d_model/h$$


* *Example:* If $d_model = 512$ and $h = 8$, each head operates on a dimensionality of $64$.

### 2. Parallel Attention (Scaled Dot-Product)

Each of the $h$ heads independently performs **Scaled Dot-Product Attention** in parallel:


![alt text](images/image-3.png)


Each head can focus on something different (e.g., Head 1 tracks *who* did the action, Head 2 tracks *when* it happened, Head 3 handles structural/grammar rules).

### 3. Concat and Project

* **Concatenation:** The outputs from all $h$ attention heads are glued together side-by-side.
* **Final Linear Layer:** The concatenated output is multiplied by a final weight matrix ($W^O$) to project it back into the original $d_{model}$ dimensions so it can be passed smoothly to the next layer (like the Feed-Forward Network).

---

## Summary of Benefits

* **Diverse Representations:** Prevents the model from getting "tunnel vision" on just one aspect of a sentence.
* **Computational Efficiency:** Because the total dimensionality is split across the heads, the total computational cost is similar to full-dimensionality single-head attention, but with much greater expressive power.

---