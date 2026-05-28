
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

---

---

# Exponentially Weighted Moving Average (EWMA) in Deep Learning
Resource: https://youtu.be/jAqVuYJ8TP8?si=EHmmHVOiq0wmqCOT

## 1. What is Exponentially Weighted Moving Average (EWMA)?

Exponentially Weighted Moving Average (EWMA or EWA) is a statistical technique used primarily to identify trends and patterns in time-series data (e.g., daily temperatures, stock prices) by smoothing out short-term fluctuations [[01:45](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=105)].

### Two Core Principles of EWMA:

1. **Recency Weighting:** More recent data points are given significantly higher weight/importance than older data points [[03:23](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=203)].
2. **Exponential Decay:** The impact or weight of any given data point decreases exponentially as time passes [[04:16](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=256)].

### Common Applications:

* Time-series and Financial Forecasting [[02:33](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=153)]
* Digital Signal Processing [[02:43](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=163)]
* **Deep Learning Optimizers:** Vital for algorithms like Momentum, RMSprop, and Adam to smooth out parameter updates [[02:53](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=173)].

---

## 2. The Mathematical Formula

The EWMA at any given time step $t$ is calculated recursively using the following equation [[05:13](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=313)]:

$$V_t = \beta V_{t-1} + (1 - \beta) \theta_t$$

Where:

* **$V_t$**: The moving average calculated at the current time step $t$ [[05:28](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=328)].
* **$V_{t-1}$**: The moving average from the *previous* time step $t-1$ [[05:38](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=338)].
* **$\theta_t$**: The actual data value observed at the current time step $t$ [[05:50](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=350)].
* **$\beta$ (Beta)**: A constant weighting parameter bounded between $0$ and $1$ ($0 \le \beta < 1$) [[06:00](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=360)].

*Note on Initialization ($V_0$): In practice, $V_0$ is typically initialized to $0$ or set equal to the first data point $\theta_1$ [[06:39](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=399)].*

---

## 3. Understanding the Impact of Beta ($\beta$)

The value assigned to $\beta$ dictates how much memory the moving average retains.

An intuitive rule of thumb is that the EWMA roughly calculates the average of the last **$\frac{1}{1 - \beta}$** days/steps of data [[08:45](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=525)].

* **High Beta ($\beta = 0.9$):**
* $\frac{1}{1 - 0.9} = 10$ days. The moving average represents the average over the past 10 days [[09:17](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=557)].
* **Behavior:** It puts a lot of weight on past trends. The resulting curve is **very smooth** and slower to adapt to sudden current changes [[11:11](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=671)].


* **Low Beta ($\beta = 0.5$):**
* $\frac{1}{1 - 0.5} = 2$ days. The moving average represents only the past 2 days [[09:47](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=587)].
* **Behavior:** It puts much more weight on the current data point ($\theta_t$). The curve becomes **highly volatile, erratic, and noisy** [[11:53](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=713)].



### Deep Learning Sweet Spot:

In deep learning optimization frameworks, **$\beta = 0.9$** is widely accepted as the standard default sweet spot [[12:48](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=768)].

---

## 4. Mathematical Proof of Recency Weighting

To see why older data points matter less, we can expand the recursive equation step-by-step for $V_4$ [[14:45](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=885)]:

$$V_4 = (1-\beta)\theta_4 + \beta(1-\beta)\theta_3 + \beta^2(1-\beta)\theta_2 + \beta^3(1-\beta)\theta_1 + \beta^4V_0$$

Because $\beta$ is a fraction less than 1 (e.g., $0.9$), raising it to higher powers ($\beta^2, \beta^3, \beta^4$) results in smaller and smaller coefficients [[15:53](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=953)]. Consequently, the most recent data point ($\theta_4$) is multiplied by the largest multiplier, while the oldest data point ($\theta_1$) is multiplied by a heavily decayed multiplier [[16:17](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=977)].

---

## 5. Python Implementation (Pandas)

In Python, the Pandas library provides a built-in method called `.ewm()` to calculate this automatically on a DataFrame Series [[17:20](https://www.youtube.com/watch?v=jAqVuYJ8TP8&t=1040)]:

```python
import pandas as pd

# 'df' contains a 'temperature' time-series column
# Note: Pandas uses 'alpha' parameter where alpha = 1 - beta
beta = 0.9
alpha = 1 - beta

df['EWMA'] = df['temperature'].ewm(alpha=alpha, adjust=False).mean()

```

---

# Optimizers in Deep Learning: SGD with Momentum

Resources: https://youtu.be/vVS4csXRlcQ?si=uKexxK9AgxxFX_gE

## 1. The Core Problem: Oscillation (Zig-Zagging)

When training a neural network using standard Stochastic Gradient Descent (SGD) or Mini-Batch Gradient Descent, the loss landscape often resembles a long, narrow ravine or valley rather than a perfect bowl.

* The slopes along the vertical axis (the sides of the ravine) are very steep.
* The slope along the horizontal axis (the path leading down to the minimum) is very gentle.

Because standard SGD only cares about the gradient at the current exact step, it gets trapped in violent **zig-zag oscillations**. It bounces back and forth rapidly between the steep walls while making agonizingly slow progress along the flat floor toward the global minimum.

If you try to speed it up by increasing the learning rate ($\alpha$), the vertical oscillations become so massive that the model completely destabilizes and blows up.

---

## 2. The Solution: What is Momentum?

**SGD with Momentum** solves this by mimicking physics. Imagine a heavy bowling ball rolling down a bumpy hill. Because of its mass and momentum, it doesn't instantly change its entire direction just because it hits a small side bump; its accumulated downward history carries it straight through.

In deep learning, Momentum forces the optimizer to **remember the direction of past updates** and use them to smooth out the current update.

* **In the vertical direction (oscillations):** The gradients keep changing signs (positive, then negative, then positive). When you average them over time, they cancel each other out, drastically reducing the bouncing.
* **In the horizontal direction (towards the minimum):** The gradients always point in the exact same forward direction. When you average them, they reinforce each other, causing the optimizer to accelerate forward.

---

## 3. The Mathematical Formula

Momentum replaces the standard gradient in the update rule with an **Exponentially Weighted Moving Average (EWMA)** of all past gradients.

### Step 1: Compute the Velocity Vector ($V_t$)

$$V_t = \beta V_{t-1} + (1 - \beta) \nabla W_t$$

### Step 2: Update the Weights

$$W_{\text{new}} = W_{\text{old}} - \alpha V_t$$

Where:

* **$\nabla W_t$**: The gradient of the loss function with respect to the weights at the current time step $t$.
* **$V_t$**: The accumulated "velocity" (the smoothed gradient direction) used to actually update the weights.
* **$\alpha$**: The learning rate.
* **$\beta$ (Momentum Coefficient)**: A hyperparameter between $0$ and $1$ that controls how much "memory" of past velocity to retain.

> **Standard Value:** Just like with EWMA, the industry standard sweet-spot value for **$\beta$ is 0.9**. This means it effectively averages the directions of the last $\frac{1}{1 - 0.9} = 10$ gradient updates.

---

## 4. Advantages of SGD with Momentum

* **Dampens Oscillations:** It smooths out the chaotic bouncing in steep, irrelevant dimensions.
* **Faster Convergence:** By accelerating along the correct directional path, it reaches the optimal minimum in far fewer steps/epochs than standard SGD.
* **Escapes Local Minima & Saddles:** Because the optimizer accumulates momentum running down a slope, the built-in "velocity" gives it the physical push needed to roll right over flat saddle points or shallow local valleys where standard gradient descent would stall out out.
* Momentum helps to reduce the noise.
* Exponential Weighted Average is used to smoothen the curve.

## 5. Disadvantage of SGD with momentum
* Extra hyperparameter is added.
* Due to momentum, the convergence becomes after becoming very close to global minima, as it requires to cancel the past avergae momentum.

---

## 6. Summary of the Evolution So Far

1. **Standard SGD:** Updates purely based on *current* gradient. High noise, violent zig-zagging in ravines, easily stuck.
2. **EWMA (Math foundation):** A formula that averages past historical data while exponentially decaying the oldest entries.
3. **SGD with Momentum:** Applies EWMA directly to the gradients. Cancels out orthogonal noise, compounds directional steps, and behaves like a ball rolling down a slope.

---

