import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the Perceptron Learning Rule
def perceptron_learning(features, labels, learning_rate=0.01, epochs=10):
    # Add a bias term to features
    bias = np.ones((features.shape[0], 1))  # Column of ones for bias
    augmented_features = np.hstack((bias, features))
    
    # Initialize weights
    weights = np.zeros(augmented_features.shape[1])  # One weight per feature plus bias

    # To store weights for visualizing decision boundaries
    weights_history = [weights.copy()]

    for epoch in range(epochs):
        for i in range(augmented_features.shape[0]):
            x_i = augmented_features[i]
            y_true = labels[i]
            
            # Predicted label (sign of the dot product)
            y_pred = np.sign(np.dot(weights, x_i))
            
            # Update weights if prediction is incorrect
            if y_pred != y_true:
                weights += learning_rate * y_true * x_i

        # Store weights after each epoch
        weights_history.append(weights.copy())

    return weights, weights_history

# Step 2: Visualize Decision Boundary Evolution
def plot_decision_boundary(features, labels, weights_history):
    plt.figure(figsize=(8, 6))
    
    # Scatter plot of data points
    plt.scatter(features[labels == 1][:, 0], features[labels == 1][:, 1], c='blue', label='Class 1 (+1)', edgecolor='k')
    plt.scatter(features[labels == -1][:, 0], features[labels == -1][:, 1], c='red', label='Class 2 (-1)', edgecolor='k')

    # Plot decision boundaries
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    x_values = np.linspace(x_min, x_max, 100)

    for epoch, weights in enumerate(weights_history):
        if epoch % 2 == 0 or epoch == len(weights_history) - 1:  # Plot every second epoch
            bias, w1, w2 = weights
            y_values = -(bias + w1 * x_values) / w2  # Decision boundary equation
            plt.plot(x_values, y_values, label=f'Epoch {epoch}')

    plt.title("Decision Boundary Evolution")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.grid(True)
    plt.show()

# Step 3: Prepare the Data
# Reuse the linearly separable data from the previous step
labels = data[:, -1]
features = data[:, :-1]

# Train the perceptron and visualize results
final_weights, weights_history = perceptron_learning(features, labels, learning_rate=0.01, epochs=10)
plot_decision_boundary(features, labels, weights_history)
