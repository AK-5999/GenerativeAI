
---

# Gradient Descent and Its Variants in Neural Networks

Resource: https://youtu.be/7z6yXpYk7sw?si=dmvvjUiih_nuyGYm

## Recall:
We know, we use different optimizers to update weights in neural network using formula.

W_new = W_old - n*(grad(l))
where,
W_new = New updated weights
W_old = old existing weights
n = learning rate
l = loss function
grad(l)= partial differentiation of loss function with respect to weights.

## 1. What is Gradient Descent?

Gradient Descent is an optimization algorithm used to minimize the **Loss Function** in machine learning and neural networks.

* It adjusts the model's parameters (weights and biases) by moving in the opposite direction of the gradient (slope) of the loss function.
* **Learning Rate ($\alpha$):** It determines the step size taken towards the minimum point. A proper learning rate helps the model converge to the optimal solution effectively.

The video highlights that when updating weights in backpropagation, there are three main variations based on **how much data is processed before calculating the gradient and updating parameters**.

---

## 2. The Three Variants of Gradient Descent

The main trade-off between these three types revolves around **Computation Time** vs. **Optimization Accuracy (Convergence)**.

### A. Batch Gradient Descent (Vanilla Gradient Descent)

* **How it works:** The algorithm takes the **entire dataset** to calculate the loss and the gradient in a single epoch.
* **Parameter Updates:** The weights and biases are updated only **once per epoch**. If you have 5 epochs, the weights are updated exactly 5 times.
* **Implementation:** It uses **Vectorization** (Matrix multiplication / dot products) via libraries like NumPy instead of using loops to process all rows simultaneously.
* **Pros & Cons:**
* **Pros:** Highly stable convergence; the loss curve decreases smoothly.
* **Cons:** It requires loading the entire dataset into RAM at once. For massive datasets (e.g., millions of rows), this causes memory overflows and stalls performance.

```
#Assume we have a dataset of 50 rows and we are training by 5 epochs.
#In BGD,
Step 1: Start Iteration for each epoch.
Step 2: calculate predicted value of y_hat variable for all rows of data in single go.
    To do this, we basically calculate the dot product between weights matrix and input data point valuse
    result will be single dimensinol array.
    Note: Dot product is efficent than for loop but both do same thing.
Step 3: Calcuate the sum average of loss function between each datapoint actual y value and y_hat predicted value.
Step 4: Update the weights using formula.
Note: Here we update the weights only 5 times.
```


### B. Stochastic Gradient Descent (SGD)

* **How it works:** The dataset is shuffled, and the algorithm picks **one random data point at a time** to calculate the prediction, loss, and gradient.
* **Parameter Updates:** The weights are updated **$N$ times per epoch** (where $N$ is the total number of rows in the dataset). For example, if you have 10 epochs and 50 rows of data, the parameters are updated $50 \times 10 = 500$ times.
* **Pros & Cons:**
* **Pros:** Much faster to converge towards a solution in terms of epochs because it updates parameters constantly. The highly random, "zig-zag" (jerkiness) path helps the algorithm **jump out of local minima** to potentially find the global minimum.
* **Cons:** Very unstable and noisy loss curve. It does not provide an exact optimal solution but rather fluctuates around an approximate solution. It is also slower per epoch in code execution due to heavy looping.

```
#Assume we have a dataset of 50 rows and we are training by 5 epochs.
#In SGD,
Step 1: Start Iteration for each epoch.
Step 2: Start sub-iteration for each row in data.
Step 3: calculate predicted value of y_hat variable for each rows of data.
    To do this, we basically calculate the sum of multiples of weights and input values.
Step 3: Calcuate the loss function between each datapoint actual y value and y_hat predicted value.
Step 4: Update the weights using formula for each data point in every iteration.
Note: Here we update the weights only epoch*rows time.
```

### C. Mini-Batch Gradient Descent

* **How it works:** This is the middle ground between Batch and Stochastic Gradient Descent. The dataset is split into **smaller groups or "batches"** (e.g., batch sizes of 32, 64, or 128).
* **Parameter Updates:** The gradient is calculated, and weights are updated after processing each mini-batch.
* **Why it's preferred:** It is the standard choice in Deep Learning. It gains the speed benefit of vectorization from Batch GD while retaining the frequent parameter updating and local-minima-escaping capability of SGD, without overloading system memory.

```
#Assume we have a dataset of 50 rows and we are training by 5 epochs.
#In SGD,
Step 1: Start Iteration for each epoch.
Step 2: Start sub-iteration for set of rows in data.
Step 3: calculate predicted value of y_hat variable for set of data.
    To do this, we basically calculate the sum of multiples of weights and input values.
Step 3: Calcuate the sum average of loss function between each datapoint actual y value and y_hat predicted value for respect set of data.
Step 4: Update the weights using formula for each set point in every iteration.
Note: Here we update the weights only epoch*batch time.
```

---

## 3. Comparative Summary Table

| Metric | Batch Gradient Descent | Stochastic Gradient Descent (SGD) | Mini-Batch Gradient Descent |
| --- | --- | --- | --- |
| **Data per Update** | Entire dataset ($N$ rows) | Exactly 1 row | A subset (Batch Size) |
| **Updates per Epoch** | 1 time | $N$ times | $\frac{N}{\text{Batch Size}}$ times |
| **Convergence Path** | Extremely smooth | Highly erratic / "noisy" | Moderately bumpy |
| **Speed per Epoch** | Fastest | Slowest (due to loops) | Moderate to Fast |
| **Memory Usage** | Very High (Requires all data in RAM) | Very Low (1 row at a time) | Low / Controlled |

---

## 4. Key Practical and Interview Insights

* **Why are Batch Sizes usually powers of 2 (32, 64, 128, etc.)?**
Using batch sizes in powers of 2 allows computers to map the data layout directly to the binary-based architecture of the RAM and GPU. This optimizes memory cache usage, making code run faster. However, it isn't a strict rule; any number will work.
* **What happens if the batch size doesn't perfectly divide the dataset?**
If you have 400 rows and set a batch size of 150, the algorithm will simply create 3 batches: two batches of 150 rows, and a final remaining batch of 100 rows. Keras and TensorFlow handle this partial final batch automatically.

---

Based on the provided video, **"Optimizers in Deep Learning | Part 1 | Complete Deep Learning Course"** by CampusX, here is a structured and copy-ready detailed summary covering the introductory concepts and core motivations behind deep learning optimizers.

---

# Optimizers in Deep Learning: Introduction & Core Challenges
Resource: https://youtu.be/iCTTnQJn50E?si=f61FMUUsFEseziBa
## 1. Context: Why Do We Need Better Optimizers?

In deep learning, training deep neural networks with numerous hidden layers is computationally expensive and time-consuming. Computer scientists focus heavily on methods to **speed up training**. While weight initialization, batch normalization, and choosing the right activation functions help, optimizing algorithms—specifically **Optimizers**—are among the most critical tools for accelerating and improving performance [[01:21](http://www.youtube.com/watch?v=iCTTnQJn50E&t=81)].

The primary goal of a neural network during training is to adjust its weights ($W$) and biases ($b$) so that its predictions ($\hat{y}$) match the real target values ($y$) as closely as possible, thereby minimizing a defined **Loss Function** [[03:35](http://www.youtube.com/watch?v=iCTTnQJn50E&t=215)].

---

## 2. The Starting Point: Conventional Gradient Descent

We start by assigning random values to the parameters (weights and biases) and gradually adjusting them toward an optimal solution where the loss is lowest [[05:08](http://www.youtube.com/watch?v=iCTTnQJn50E&t=308)]. This process relies on **Gradient Descent**, which uses a fundamental update rule:

$$\text{New Weight} = \text{Old Weight} - (\alpha \times \text{Gradient})$$

Where $\alpha$ is the **Learning Rate**, controlling the size of the steps taken toward the minimum [[07:29](http://www.youtube.com/watch?v=iCTTnQJn50E&t=449)]. There are three established variants of this conventional approach based on how much data is evaluated before an update [[08:06](http://www.youtube.com/watch?v=iCTTnQJn50E&t=486)]:

1. **Batch Gradient Descent:** Uses the entire dataset to compute the loss and gradient, updating the parameters just once per epoch.
2. **Stochastic Gradient Descent (SGD):** Updates the weights immediately after looking at *every single data point*, leading to highly frequent updates.
3. **Mini-Batch Gradient Descent:** Splits data into smaller subgroups (batches) and updates parameters after each batch, serving as a balanced middle ground.

---

## 3. Five Key Challenges of Conventional Gradient Descent

Despite these variants, conventional gradient descent faces 5 major challenges that make it either too slow or prone to getting stuck with sub-optimal results [[20:43](http://www.youtube.com/watch?v=iCTTnQJn50E&t=1243)]:

### I. Hard-to-Tune Learning Rate ($\alpha$)

* Setting the right learning rate is incredibly difficult.
* If $\alpha$ is **too small**, the algorithm takes minuscule steps, resulting in painfully slow convergence [[12:08](http://www.youtube.com/watch?v=iCTTnQJn50E&t=728)].
* If $\alpha$ is **too large**, the steps overshoot the minimum, causing unstable training behavior where the model oscillates wildly or fails to converge altogether [[12:22](http://www.youtube.com/watch?v=iCTTnQJn50E&t=742)].

### II. Rigid Learning Rate Scheduling

* A technique called *Learning Rate Scheduling* was introduced to dynamically reduce the learning rate over time.
* However, these schedules must be predefined manually before training begins. Because they don't adapt directly to the nature of the dataset during training, they fail to work universally well across different data types [[15:00](http://www.youtube.com/watch?v=iCTTnQJn50E&t=900)].

### III. Uniform Learning Rate across All Dimensions

* Neural networks have millions of parameters, meaning optimization occurs in a massive multi-dimensional space.
* Conventional gradient descent applies the **exact same learning rate to every parameter** (all weights and biases) [[16:43](http://www.youtube.com/watch?v=iCTTnQJn50E&t=1003)]. This is restrictive because some dimensions/parameters might require faster, larger adjustments, while others need slow, cautious stepping [[17:12](http://www.youtube.com/watch?v=iCTTnQJn50E&t=1032)].

### IV. Getting Stuck in Local Minima

* The cost landscapes of deep neural networks are complex and non-convex, containing multiple valleys (**Local Minima**) [[18:10](http://www.youtube.com/watch?v=iCTTnQJn50E&t=1090)].
* While the ultimate objective is to find the lowest valley (**Global Minimum**), conventional gradient descent algorithms can easily get trapped inside a local minimum with no mechanism to climb or jump out, leaving the model with sub-optimal performance [[18:39](http://www.youtube.com/watch?v=iCTTnQJn50E&t=1119)].

### V. Trapped by Saddle Points

* A **Saddle Point** is a region where the slope curves up in one direction but curves down in another [[19:33](http://www.youtube.com/watch?v=iCTTnQJn50E&t=1173)].
* This creates a broad, plateau-like region where the gradients drop effectively to zero. When the gradient is zero, the update formula yields no change ($\text{New Weight} = \text{Old Weight}$), causing the network to stall and stop learning completely [[20:00](http://www.youtube.com/watch?v=iCTTnQJn50E&t=1200)].

---

## 4. Advanced Optimizers to Be Explored

To overcome these five limitations, advanced optimization algorithms introduce small, clever mathematical tweaks over basic gradient descent. The upcoming videos in this series will cover five essential deep learning optimizers [[21:10](http://www.youtube.com/watch?v=iCTTnQJn50E&t=1270)]:

1. **Momentum**
2. **NAG (Nesterov Accelerated Gradient)**
3. **AdaGrad**
4. **RMSprop**
5. **Adam** (the most widely used optimizer in modern Deep Learning)

*Note: To understand how these advanced optimizers function, the next foundational concept to learn is **Exponentially Weighted Moving Averages (EWMA)**, which dictates how past gradients influence current updates [[21:57](http://www.youtube.com/watch?v=iCTTnQJn50E&t=1317)].*