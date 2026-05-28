Here are the detailed, easy-to-understand study notes compiled from Krish Naik's Deep Learning series covering 
* **"Introduction to Deep Learning"** (Tutorial 1)
* **"How does a Neural Network Work"** (Tutorial 2).

---

# Part 1: Introduction to Deep Learning

### What is Deep Learning?

Deep Learning (DL) is a specialized subset of **Machine Learning (ML)**, which itself is a subset of **Artificial Intelligence (AI)**.

* **Artificial Intelligence:** The broad concept of creating smart machines capable of performing tasks that typically require human intelligence.
* **Machine Learning:** Giving computers the ability to learn from data without being explicitly programmed.
* **Deep Learning:** Using multi-layered artificial neural networks to mimic the human brain’s ability to learn from massive amounts of unstructured data.

### Why Did Deep Learning Become So Popular?

While Machine Learning has been around for a long time, Deep Learning exploded in popularity recently due to two major factors:

1. **The Explosion of Data:** Machine Learning algorithms reach a plateau in performance after a certain amount of data. No matter how much more data you feed them, they don't get significantly smarter. Deep Learning models, however, keep getting better and more accurate as you feed them more data.
2. **Hardware Advancements (GPUs and TPUs):** Neural networks require massive amounts of mathematical computations simultaneously. The rise of powerful Graphics Processing Units (GPUs) and Tensor Processing Units (TPUs) made it possible to train complex models in hours instead of years.

### The Biological Inspiration: The Biological Neuron

Deep Learning is inspired by how our own brains process information. A biological neuron consists of three main parts:

* **Dendrites:** The receivers. They gather incoming signals (electrical impulses) from other neurons.
* **Soma (Cell Body):** The processor. It processes the incoming signals and decides what to do.
* **Axon:** The transmitter. It carries the processed signal away toward the next neuron.

Artificial Neural Networks try to copy this exact structure mathematically.

---

# Part 2: How Does an Artificial Neural Network (ANN) Work?

An Artificial Neural Network mimics the brain using layers of interconnected virtual "neurons" (often called nodes).

### 1. The Structure of a Neural Network

A basic neural network is divided into three types of layers:

* **Input Layer:** This is where your raw data enters the network (e.g., house size, number of rooms, etc.). Each feature of your data is represented by one node.
* **Hidden Layer(s):** The "magic" happens here. This layer extracts features and patterns from the data. A network can have one hidden layer (shallow network) or many hidden layers (deep network).
* **Output Layer:** This gives the final prediction or decision (e.g., the predicted price of the house, or whether an image is a cat or a dog).

---

### 2. Inside a Single Artificial Neuron (Perceptron)

To understand how the entire network functions, we have to look inside just one node of a hidden or output layer. Every single node performs a simple two-step mathematical calculation:

#### Step A: Summation (Linear Combination)

The neuron takes all the inputs ($x$) coming into it, multiplies each input by a specific weight ($w$), adds them all together, and finally adds a "bias" ($b$).

Mathematically, it looks like this:


$$Z = (x_1 \cdot w_1) + (x_2 \cdot w_2) + \dots + (x_n \cdot w_n) + b$$

* **Weights ($w$):** Think of weights as the "importance" of an input. If a specific input is highly relevant to making a correct prediction, its weight will be high.
* **Bias ($b$):** The bias allows the neuron to shift the output up or down, helping the model better fit patterns in the data.

#### Step B: The Activation Function

The raw result $Z$ is just a simple linear equation. Real-world data is complex and non-linear. Therefore, the result $Z$ is passed through an **Activation Function** (represented as $f(Z)$).

* **Purpose:** It adds non-linearity to the network, allowing it to learn complex patterns (like curves, shapes, and complex boundaries).
* **Common Examples:**
* **Sigmoid:** Squashes the output value to a tight range between 0 and 1 (great for binary classification like Yes/No).
* **ReLU (Rectified Linear Unit):** Keeps positive values exactly as they are, but turns any negative value into 0. This is the most popular function used in hidden layers because it is incredibly fast to compute.



---

### 3. The Complete Flow: How the Network Learns

A neural network doesn't start off smart. It learns through a continuous loop of two main processes:

#### Process 1: Forward Propagation

1. Data enters through the **Input Layer**.
2. It travels forward through the **Hidden Layers**, undergoing the summation and activation functions at every single node.
3. The **Output Layer** generates a final prediction ($\hat{y}$).

#### Process 2: Backward Propagation (The Learning Phase)

1. **Calculate the Loss (Error):** The network compares its predicted output ($\hat{y}$) with the actual true value ($y$) to calculate how wrong it was. This error is measured using a **Loss Function**.
2. **Send Feedback Backwards:** The error is sent backward through the network.
3. **Update the Weights:** Using an optimization algorithm (like **Gradient Descent**), the network adjusts all the weights and biases slightly to reduce the error.

This entire loop (Forward Propagation $\rightarrow$ Error Calculation $\rightarrow$ Backward Propagation) repeats thousands of times over the data until the error is minimized and the network becomes highly accurate.

---