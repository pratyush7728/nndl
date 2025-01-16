import numpy as np

# Input data and sigmoid function
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
sigmoid = lambda z: 1 / (1 + np.exp(-z))

# Train the perceptron
def train_perceptron(targets, lr=0.1, epochs=5000):
    w1, w2, bias = 0.8, 0.9, 0.25
    for _ in range(epochs):
        for i in range(4):
            z = w1 * x[i][0] + w2 * x[i][1] + bias
            error = targets[i] - sigmoid(z)
            w1 += lr * error * x[i][0]
            w2 += lr * error * x[i][1]
            bias += lr * error
    return w1, w2, bias

# Test the perceptron
def test_perceptron(w1, w2, bias):
    for i in range(4):
        z = w1 * x[i][0] + w2 * x[i][1] + bias
        result = sigmoid(z)
        predicted = 1 if result >= 0.5 else 0
        print(f"Input: {x[i]}, Output: {result:.4f}, Predicted: {predicted}")

# Train and test for OR gate
print("OR Gate:")
w1, w2, bias = train_perceptron([0, 1, 1, 1])
test_perceptron(w1, w2, bias)

print("\nAND Gate:")
# Train and test for AND gate
w1, w2, bias = train_perceptron([0, 0, 0, 1])
test_perceptron(w1, w2, bias)
