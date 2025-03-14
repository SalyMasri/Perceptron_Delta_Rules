import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate Gaussian Distributed Data
np.random.seed(42)  # For reproducibility
mean_class_1, cov_class_1 = [2, 2], [[1, 0.5], [0.5, 1]]
mean_class_2, cov_class_2 = [-2, -2], [[1, 0.5], [0.5, 1]]

class_1 = np.random.multivariate_normal(mean=mean_class_1, cov=cov_class_1, size=50)
class_2 = np.random.multivariate_normal(mean=mean_class_2, cov=cov_class_2, size=50)

labels_1, labels_2 = np.ones((50, 1)), -1 * np.ones((50, 1))
data_class_1 = np.hstack((class_1, labels_1))
data_class_2 = np.hstack((class_2, labels_2))
data = np.vstack((data_class_1, data_class_2))

features, labels = data[:, :-1], data[:, -1]

# Step 2: Define Methods (Perceptron, Delta Rule Batch/Sequential)

# Perceptron Learning Rule
def perceptron_learning(features, labels, learning_rate=0.01, epochs=10):
    bias = np.ones((features.shape[0], 1))
    augmented_features = np.hstack((bias, features))
    weights = np.zeros(augmented_features.shape[1])
    weights_history = [weights.copy()]
    errors_by_epoch = []

    for epoch in range(epochs):
        total_errors = 0
        for i in range(augmented_features.shape[0]):
            x_i, y_true = augmented_features[i], labels[i]
            y_pred = np.sign(np.dot(weights, x_i))
            if y_pred != y_true:
                weights += learning_rate * y_true * x_i
                total_errors += 1
        errors_by_epoch.append(total_errors)
        weights_history.append(weights.copy())
    return weights, weights_history, errors_by_epoch

# Delta Rule (Batch)
def delta_rule_batch(features, labels, learning_rate=0.01, epochs=10):
    bias = np.ones((features.shape[0], 1))
    augmented_features = np.hstack((bias, features))
    weights = np.zeros(augmented_features.shape[1])
    weights_history = [weights.copy()]
    errors_by_epoch = []

    for epoch in range(epochs):
        predictions = np.dot(augmented_features, weights)
        errors = labels - predictions
        squared_error = np.sum(errors ** 2)
        errors_by_epoch.append(squared_error)
        weights += learning_rate * np.dot(errors.T, augmented_features)
        weights_history.append(weights.copy())
    return weights, weights_history, errors_by_epoch

# Delta Rule (Sequential)
def delta_rule_sequential(features, labels, learning_rate=0.01, epochs=10):
    bias = np.ones((features.shape[0], 1))
    augmented_features = np.hstack((bias, features))
    weights = np.zeros(augmented_features.shape[1])
    weights_history = [weights.copy()]
    errors_by_epoch = []

    for epoch in range(epochs):
        squared_error = 0
        for i in range(augmented_features.shape[0]):
            x_i, y_true = augmented_features[i], labels[i]
            y_pred = np.dot(weights, x_i)
            error = y_true - y_pred
            squared_error += error ** 2
            weights += learning_rate * error * x_i
        errors_by_epoch.append(squared_error)
        weights_history.append(weights.copy())
    return weights, weights_history, errors_by_epoch

# Step 3: Plotting Functions
def plot_decision_boundary(features, labels, weights_history):
    plt.figure(figsize=(8, 6))
    plt.scatter(features[labels == 1][:, 0], features[labels == 1][:, 1], c='blue', label='Class 1 (+1)', edgecolor='k')
    plt.scatter(features[labels == -1][:, 0], features[labels == -1][:, 1], c='red', label='Class 2 (-1)', edgecolor='k')
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    x_values = np.linspace(x_min, x_max, 100)

    for epoch, weights in enumerate(weights_history):
        if epoch % 2 == 0 or epoch == len(weights_history) - 1:
            bias, w1, w2 = weights
            y_values = -(bias + w1 * x_values) / w2
            plt.plot(x_values, y_values, label=f"Epoch {epoch}")

    plt.title("Decision Boundary Evolution")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.grid(True)
    plt.show()  # Display the plot


def plot_error_by_epoch(errors, method_name):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(errors) + 1), errors, label=f"{method_name} Errors")
    plt.title(f"{method_name} Error by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()  # Display the plot


# Step 4: Train Models and Save Plots
# Train models
# Step 4: Train Models and Plot Results
# Train models
perceptron_weights, perceptron_history, perceptron_errors = perceptron_learning(features, labels)
batch_weights, batch_history, batch_errors = delta_rule_batch(features, labels)
sequential_weights, sequential_history, sequential_errors = delta_rule_sequential(features, labels)

# Plot decision boundary evolution for Perceptron
print("Perceptron Decision Boundary Evolution:")
plot_decision_boundary(features, labels, perceptron_history)

# Plot decision boundary evolution for Delta Rule (Batch)
print("Batch Delta Rule Decision Boundary Evolution:")
plot_decision_boundary(features, labels, batch_history)

# Plot decision boundary evolution for Delta Rule (Sequential)
print("Sequential Delta Rule Decision Boundary Evolution:")
plot_decision_boundary(features, labels, sequential_history)

# Plot error by epoch for Perceptron
print("Perceptron Error by Epoch:")
plot_error_by_epoch(perceptron_errors, "Perceptron Learning")

# Plot error by epoch for Delta Rule (Batch)
print("Batch Delta Rule Error by Epoch:")
plot_error_by_epoch(batch_errors, "Delta Rule (Batch)")

# Plot error by epoch for Delta Rule (Sequential)
print("Sequential Delta Rule Error by Epoch:")
plot_error_by_epoch(sequential_errors, "Delta Rule (Sequential)")

# Save error by epoch plot

# Save individual error plots
plot_error_by_epoch(perceptron_errors, "Perceptron Learning", "perceptron_error_by_epoch.pdf")
plot_error_by_epoch(batch_errors, "Delta Rule (Batch)", "batch_delta_rule_error_by_epoch.pdf")
plot_error_by_epoch(sequential_errors, "Delta Rule (Sequential)", "sequential_delta_rule_error_by_epoch.pdf")




# Define a range of learning rates
learning_rates = [0.01, 0.1, 0.5, 1.0]
epochs = 20  # Increase epochs for better observation

# Store results for comparison
results = {
    "Perceptron": {},
    "Delta Batch": {},
    "Delta Sequential": {}
}

# Train models for each learning rate
for lr in learning_rates:
    perceptron_weights, perceptron_history, perceptron_errors = perceptron_learning(features, labels, learning_rate=lr, epochs=epochs)
    batch_weights, batch_history, batch_errors = delta_rule_batch(features, labels, learning_rate=lr, epochs=epochs)
    sequential_weights, sequential_history, sequential_errors = delta_rule_sequential(features, labels, learning_rate=lr, epochs=epochs)
    
    # Store errors
    results["Perceptron"][lr] = perceptron_errors
    results["Delta Batch"][lr] = batch_errors
    results["Delta Sequential"][lr] = sequential_errors



def plot_combined_learning_curves(results, method_name, filename):
    plt.figure(figsize=(10, 8))
    for lr, errors in results[method_name].items():
        plt.plot(range(1, len(errors) + 1), errors, label=f"LR={lr}")
    plt.title(f"{method_name} Learning Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, format='pdf')
    plt.close()

# Plot combined curves for each method
plot_combined_learning_curves(results, "Perceptron", "perceptron_learning_curves.pdf")
plot_combined_learning_curves(results, "Delta Batch", "batch_learning_curves.pdf")
plot_combined_learning_curves(results, "Delta Sequential", "sequential_learning_curves.pdf")








def perceptron_without_bias_fixed(features, labels, learning_rate=0.01, epochs=10):
    weights = np.zeros(features.shape[1])  # No bias term
    weights_history = [weights.copy()]
    errors_by_epoch = []

    for epoch in range(epochs):
        total_errors = 0
        for i in range(features.shape[0]):
            x_i, y_true = features[i], labels[i]
            y_pred = np.sign(np.dot(weights, x_i))
            if y_pred != y_true:
                weights += learning_rate * y_true * x_i
                total_errors += 1
        errors_by_epoch.append(total_errors)
        weights_history.append(weights.copy())
    return weights, weights_history, errors_by_epoch







# Create symmetric data
# Create symmetric data for better performance without bias
mean_class_1, mean_class_2 = [1, 1], [-1, -1]
cov_class = [[0.5, 0.1], [0.1, 0.5]]

class_1 = np.random.multivariate_normal(mean=mean_class_1, cov=cov_class, size=50)
class_2 = np.random.multivariate_normal(mean=mean_class_2, cov=cov_class, size=50)

# Combine features and labels
labels_1 = np.ones((50,))
labels_2 = -1 * np.ones((50,))
features_symmetric = np.vstack((class_1, class_2))
labels_symmetric = np.hstack((labels_1, labels_2))





# Train the perceptron without bias
weights_no_bias, history_no_bias, errors_no_bias = perceptron_without_bias_fixed(features_symmetric, labels_symmetric)

# Plot decision boundary
def plot_decision_boundary_no_bias(features, labels, weights_history, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(features[labels == 1][:, 0], features[labels == 1][:, 1], c='blue', label='Class 1 (+1)', edgecolor='k')
    plt.scatter(features[labels == -1][:, 0], features[labels == -1][:, 1], c='red', label='Class 2 (-1)', edgecolor='k')
    x_values = np.linspace(features[:, 0].min() - 1, features[:, 0].max() + 1, 100)

    for epoch, weights in enumerate(weights_history):
        if epoch % 2 == 0 or epoch == len(weights_history) - 1:
            w1, w2 = weights
            y_values = -(w1 * x_values) / w2  # Decision boundary passes through the origin
            plt.plot(x_values, y_values, label=f"Epoch {epoch}")

    plt.title("Decision Boundary Evolution Without Bias")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, format='pdf')
    plt.close()

# Save the plot
plot_decision_boundary_no_bias(features_symmetric, labels_symmetric, history_no_bias, 'decision_boundary_no_bias.pdf')
