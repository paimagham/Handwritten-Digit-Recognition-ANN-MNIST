# Handwritten Digit Recognition — Neural Network from Scratch

A neural network built **using only NumPy** — no PyTorch, TensorFlow, or Keras — that learns to read handwritten digits (0 through 9). Every part of the network, including the learning process itself, is written by hand to show exactly how a neural network works under the hood.

## What is MNIST?

MNIST is one of the most famous datasets in machine learning — often called the "Hello World" of deep learning. It contains **70,000 images of handwritten digits** (0–9), each written by a different person and scanned into a small **28×28 pixel** grayscale image.

Because the handwriting comes from thousands of different people, the digits vary in shape, thickness, and style — which makes it a great test of whether a computer can learn to recognize patterns the way humans do. The task is simple to state but not trivial to solve: given an image of a handwritten number, correctly identify which digit (0–9) it is.

## What This Project Does

This project teaches a computer to recognize those handwritten digits by building a neural network from the ground up:

1. **Loads** 70,000 digit images from MNIST
2. **Prepares** the data by normalizing pixel values and splitting into training and testing sets
3. **Builds** a neural network by hand with three layers
4. **Trains** the network by showing it thousands of examples and letting it learn from its mistakes
5. **Evaluates** how well it recognizes digits it has never seen before

## How the Network is Built
- **Input layer (784 neurons):** one neuron for each pixel in the 28×28 image
- **Hidden layer (128 neurons):** finds patterns like edges, curves, and loops
- **Output layer (10 neurons):** one for each possible digit, giving the final prediction

The interesting part is that **everything is coded manually** — the forward pass, the backpropagation (how the network learns from mistakes), and the weight updates. There is no machine learning library doing the math for me; it's all written using NumPy.

## Key Pieces Explained Simply

| Part | What it does |
|------|--------------|
| **Sigmoid** | Squashes numbers into a 0–1 range, like a dimmer switch |
| **Softmax** | Turns the output into probabilities that add up to 100% |
| **Forward pass** | Sends an image through the network to make a guess |
| **Backpropagation** | Compares the guess to the right answer and adjusts the network to do better next time |
| **Training loop** | Repeats this process 100 times so the network keeps improving |

## Results

As training progresses, the network's mistakes go down and its accuracy goes up. The project generates four visualizations to show what's happening:

| Image | What it shows |
|-------|---------------|
| `mnist_samples.png` | Example handwritten digits from the dataset |
| `training_progress.png` | Loss going down and accuracy going up over time |
| `confusion_matrix.png` | Which digits the network sometimes mixes up |
| `sample_predictions.png` | Real predictions, with correct ones in green and mistakes in red |

## Tech Stack

- **Python** — the core language used to build the entire neural network and training pipeline
- **NumPy** — powers all the math behind the network: matrix multiplication, weight updates, activation functions, and the backpropagation calculations, all implemented manually without any deep learning framework
- **scikit-learn** — used only for loading the MNIST dataset, splitting it into training and testing sets, standardizing the pixel values, and computing evaluation metrics (accuracy, confusion matrix, classification report)
- **Matplotlib** — creates the visualizations, including the training loss curve, validation accuracy curve, and sample digit images
- **Seaborn** — used to plot the confusion matrix as a clean, easy-to-read heatmap showing which digits the network confuses

## How to Run

```bash
pip install numpy scikit-learn matplotlib seaborn
python ann_mnist.py
```

## Why I Built This

I wanted to truly understand how neural networks learn, not just call a library function. Writing the math by hand — especially backpropagation — gave me a solid foundation that later helped me work on more advanced topics like reinforcement learning, where understanding how models learn from feedback is essential.
