# =============================================================================
# Handwritten Digit Recognition — Artificial Neural Network from Scratch using NumPy
# =============================================================================
# A fully-connected neural network built using only NumPy — no PyTorch,
# TensorFlow, or Keras. Forward propagation, backpropagation, and gradient
# descent are all implemented by hand to show how a neural network learns
# under the hood.
#
# Architecture:  784 (input) -> 128 (hidden, sigmoid) -> 10 (output, softmax)
# Dataset:       MNIST handwritten digits (0-9)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =============================================================================
# 1. Load the MNIST dataset
# =============================================================================
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = np.array(mnist.data, dtype=float)
y = np.array(mnist.target, dtype=int)
print(f"Dataset loaded: {X.shape[0]:,} images, {X.shape[1]} pixels each "
      f"({len(np.unique(y))} classes)")


# =============================================================================
# 2. Prepare the data
# =============================================================================
# Use a subset for faster training on CPU
X_subset = X[:10000]
y_subset = y[:10000]

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.2, random_state=42
)

# Standardize pixel values (mean 0, unit variance) for stable training
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Training samples: {X_train.shape[0]:,} | Test samples: {X_test.shape[0]:,}")


# =============================================================================
# 3. Visualize sample digits
# =============================================================================
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Sample Handwritten Digits from MNIST', fontsize=16, fontweight='bold')
for i, ax in enumerate(axes.flat):
    # Reshape the flat 784-value vector back into a 28x28 image
    ax.imshow(X_subset[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Label: {y_subset[i]}', fontsize=12, fontweight='bold')
    ax.axis('off')
plt.tight_layout()
plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
plt.show()


# =============================================================================
# 4. The Neural Network (built entirely from scratch)
# =============================================================================
class NeuralNetwork:
    """
    A simple fully-connected neural network with one hidden layer.

        Input (784) -> Hidden (128, sigmoid) -> Output (10, softmax)

    Forward propagation, backpropagation, and gradient descent are all
    implemented by hand using NumPy.
    """

    def __init__(self, input_size, hidden_size, output_size, seed=42):
        rng = np.random.default_rng(seed)
        # Xavier-style small random initialization (suited to sigmoid activations)
        self.W1 = rng.standard_normal((input_size, hidden_size)) * np.sqrt(1.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.standard_normal((hidden_size, output_size)) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    # ---- Activation functions ------------------------------------------------
    def sigmoid(self, z):
        """Squash any value into the range (0, 1)."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, a):
        """Derivative of sigmoid, given the activation a = sigmoid(z)."""
        return a * (1.0 - a)

    def softmax(self, z):
        """Convert raw scores into probabilities that sum to 1."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # ---- Forward pass --------------------------------------------------------
    def forward(self, X):
        """Pass the input through the network to produce predictions."""
        self.z1 = X @ self.W1 + self.b1     # input -> hidden
        self.a1 = self.sigmoid(self.z1)     # hidden activation
        self.z2 = self.a1 @ self.W2 + self.b2  # hidden -> output
        self.a2 = self.softmax(self.z2)     # output probabilities
        return self.a2

    # ---- Backward pass -------------------------------------------------------
    def backward(self, X, y, output, learning_rate):
        """
        Backpropagation: compute gradients via the chain rule and update
        the weights with gradient descent.
        """
        m = X.shape[0]

        # One-hot encode the labels (e.g. 3 -> [0,0,0,1,0,0,0,0,0,0])
        y_one_hot = np.zeros((m, 10))
        y_one_hot[np.arange(m), y] = 1

        # Gradient at the output layer (softmax + cross-entropy)
        dz2 = output - y_one_hot
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Gradient at the hidden layer
        dz1 = (dz2 @ self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Gradient descent weight updates
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    # ---- Training loop -------------------------------------------------------
    def train(self, X, y, X_val, y_val, epochs, learning_rate):
        train_losses, val_accuracies = [], []
        m = X.shape[0]
        y_one_hot = np.zeros((m, 10))
        y_one_hot[np.arange(m), y] = 1

        print(f"\nTraining for {epochs} epochs (lr={learning_rate})...")
        for epoch in range(epochs):
            output = self.forward(X)

            # Cross-entropy loss
            loss = -np.mean(np.sum(y_one_hot * np.log(output + 1e-8), axis=1))
            train_losses.append(loss)

            self.backward(X, y, output, learning_rate)

            val_acc = accuracy_score(y_val, self.predict(X_val))
            val_accuracies.append(val_acc)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        return train_losses, val_accuracies

    def predict(self, X):
        """Return the predicted digit (0-9) for each input."""
        return np.argmax(self.forward(X), axis=1)


# =============================================================================
# 5. Train the network
# =============================================================================
nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
train_losses, val_accuracies = nn.train(
    X_train, y_train, X_test, y_test, epochs=100, learning_rate=0.1
)


# =============================================================================
# 6. Evaluate
# =============================================================================
y_pred = nn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal test accuracy: {accuracy*100:.2f}%\n")
print("Classification report:\n")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))


# =============================================================================
# 7. Training curves
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(train_losses, linewidth=2, color='#e74c3c')
ax1.set_title('Training Loss Over Time\n(Lower is Better)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(val_accuracies, linewidth=2, color='#27ae60')
ax2.set_title('Validation Accuracy Over Time\n(Higher is Better)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 1])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
plt.show()


# =============================================================================
# 8. Confusion matrix
# =============================================================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix\n(Diagonal = Correct Predictions)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Digit', fontsize=12, fontweight='bold')
plt.ylabel('True Digit', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()


# =============================================================================
# 9. Sample predictions
# =============================================================================
fig, axes = plt.subplots(3, 5, figsize=(14, 8))
fig.suptitle('Sample Predictions (Green = Correct, Red = Wrong)', fontsize=16, fontweight='bold')
# Un-standardize just for display so the digits look natural
X_test_display = scaler.inverse_transform(X_test)
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test_display[i].reshape(28, 28), cmap='gray')
    color = 'green' if y_pred[i] == y_test[i] else 'red'
    ax.set_title(f'True: {y_test[i]} | Pred: {y_pred[i]}',
                 fontsize=11, fontweight='bold', color=color)
    ax.axis('off')

plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nDone. Architecture: 784 -> 128 -> 10 | Final accuracy: {accuracy*100:.2f}%")
