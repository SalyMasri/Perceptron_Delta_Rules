import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# 1) Data Generation Utilities
##############################################################################

def generate_data_class(mean, sigma, n):
    """
    Generates 2D data (shape (2,n)) from a normal distribution
    with center = mean (2D) and standard deviation sigma (scalar).
    """
    # mean is a 2D vector (e.g. [1.0, 0.5])
    # sigma is a scalar that we apply to each dimension
    # n is the number of points
    data = np.random.randn(2, n) * sigma + np.array(mean).reshape(2,1)
    return data

def shuffle_data(dataA, dataB):
    """
    Shuffles the columns (samples) of data from two classes.
    dataA and dataB each have shape (2,n).
    Returns X (2,2n) and T (1,2n) with random ordering.
    """
    # Stack horizontally
    X = np.hstack((dataA, dataB))
    # Labels: +1 for classA, -1 for classB
    T = np.hstack((np.ones(dataA.shape[1]), -1 * np.ones(dataB.shape[1])))
    
    # Shuffle
    idx = np.random.permutation(X.shape[1])
    X = X[:, idx]
    T = T[idx]
    
    return X, T

def add_bias_term(X):
    """
    Adds a row of ones to X for bias. 
    If X has shape (2, N), the output will have shape (3, N).
    """
    return np.vstack((X, np.ones((1, X.shape[1]))))

##############################################################################
# 2) Single-Layer Perceptron: Activation & Performance
##############################################################################

def perceptron_activation(net_out):
    """
    Hard threshold at zero -> returns +1 or -1
    """
    return np.where(net_out >= 0, 1, -1)

def compute_accuracy(X, T, W):
    """
    Compute classification accuracy (fraction correctly labeled).
    X has shape (d, N), T has shape (N,) of +1/-1, W has shape (d,).
    """
    # net output
    net = np.dot(W, X)
    pred = perceptron_activation(net)
    correct = np.sum(pred == T)
    return correct / float(X.shape[1])

def mean_squared_error(X, T, W):
    """
    Just a convenience for the delta rule's error tracking:
    MSE = average of (net - target)^2
    """
    net = np.dot(W, X)  # shape (N,) 
    diff = net - T
    return np.mean(diff**2)

##############################################################################
# 3) Training Algorithms
##############################################################################

def perceptron_training_online(X, T, eta=0.01, epochs=20):
    """
    Perceptron learning rule in ONLINE (sequential) mode.
    X: shape (d, N) including bias row
    T: shape (N,) in {+1, -1}
    Returns: (W, acc_history) 
             W is the final weight vector of shape (d,)
             acc_history is a list of accuracies across epochs
    """
    d, N = X.shape
    # Initialize W with small random values
    W = 0.01 * np.random.randn(d)
    
    acc_history = []
    for e in range(epochs):
        # Shuffle samples for each epoch (optional, but often recommended)
        idx = np.random.permutation(N)
        for i in idx:
            x_i = X[:, i]
            t_i = T[i]
            # Net output
            net_i = np.dot(W, x_i)
            y_i = 1 if net_i >= 0 else -1
            # Perceptron update if misclassified
            if y_i != t_i:
                W += eta * t_i * x_i
        # Track accuracy
        acc = compute_accuracy(X, T, W)
        acc_history.append(acc)
    return W, acc_history

def delta_rule_training_online(X, T, eta=0.01, epochs=20):
    """
    Delta (Widrow-Hoff) rule in ONLINE (sequential) mode.
    At each sample, we do:
       W <- W - eta * (y - t) * x
    where y = W^T x (no step). We'll still interpret sign(y) as the label.
    X: shape (d,N)
    T: shape (N,) in {+1, -1}
    Returns: (W, mse_history)
    """
    d, N = X.shape
    W = 0.01 * np.random.randn(d)
    
    mse_history = []
    for e in range(epochs):
        # Shuffle samples
        idx = np.random.permutation(N)
        for i in idx:
            x_i = X[:, i]
            t_i = T[i]
            y_i = np.dot(W, x_i)
            # Delta rule update
            error_i = (y_i - t_i)
            W -= eta * error_i * x_i
        
        # Compute MSE for monitoring
        mse = mean_squared_error(X, T, W)
        mse_history.append(mse)
    return W, mse_history

def delta_rule_training_batch(X, T, eta=0.001, epochs=20):
    """
    Delta (Widrow-Hoff) rule in BATCH mode.
    W <- W - eta * (W X - T) X^T   (vectorized)
    X shape (d,N), T shape (N,)
    """
    d, N = X.shape
    W = 0.01 * np.random.randn(d)
    
    mse_history = []
    for e in range(epochs):
        net = np.dot(W, X)  # shape (N,)
        diff = net - T      # shape (N,)
        grad = np.dot(X, diff)  # shape (d,)
        W -= eta * grad
        
        # track error
        mse = mean_squared_error(X, T, W)
        mse_history.append(mse)
    return W, mse_history

##############################################################################
# 4) Plotting Utilities
##############################################################################

def plot_data_and_boundary(X, T, W, title=""):
    """
    Plots 2D data with classes T = +1 and T = -1, plus the line W x = 0.
    X has shape (3, N) if it includes a bias row.
    W has shape (3,).
    """
    plt.figure()
    # Separate points
    classA = X[:2, T==1]
    classB = X[:2, T==-1]
    plt.scatter(classA[0,:], classA[1,:], color='b', label='Class +1')
    plt.scatter(classB[0,:], classB[1,:], color='r', label='Class -1')
    
    # Decision boundary: solve W0*x + W1*y + W2 = 0 for y
    # => y = -(W2 + W0*x) / W1
    x_vals = np.linspace(np.min(X[0,:]), np.max(X[0,:]), 100)
    if abs(W[1]) > 1e-5:
        y_vals = -(W[0]*x_vals + W[2]) / W[1]
        plt.plot(x_vals, y_vals, 'k-', label='Decision boundary')
    
    plt.title(title)
    plt.legend()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.show()

def plot_learning_curves(curve1, curve2=None, label1="Method1", label2="Method2", ylabel="Accuracy or MSE"):
    """
    Utility to compare one or two learning curves side-by-side.
    """
    plt.figure()
    plt.plot(curve1, label=label1)
    if curve2 is not None:
        plt.plot(curve2, label=label2)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


# Function to overlay learning curves for comparison
def plot_combined_learning_curves(acc_perc, mse_delta_online, mse_delta_batch, title="Learning Curve Comparison"):
    """
    Plot combined learning curves for Perceptron accuracy and Delta Rule (online and batch) MSE.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(acc_perc, label="Perceptron Accuracy (Online)", color='blue', linestyle='-')
    plt.plot(mse_delta_online, label="Delta Rule MSE (Online)", color='red', linestyle='--')
    plt.plot(mse_delta_batch, label="Delta Rule MSE (Batch)", color='green', linestyle='-.')
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy / MSE")
    plt.legend()
    plt.grid(True)
    plt.show()

##############################################################################
# 5) Demonstration "main"
##############################################################################

def main():
    np.random.seed(42)  # for reproducible results

    #---------------------------------------------------------------------------
    # Part 3.1.1: Generate linearly-separable data
    #---------------------------------------------------------------------------
    n = 100
    mA = [1.0, 0.5]; sigmaA = 0.5
    mB = [-1.0, 0.0]; sigmaB = 0.5

    classA = generate_data_class(mA, sigmaA, n)
    classB = generate_data_class(mB, sigmaB, n)

    # Plot initial class distributions
    plt.figure()
    plt.scatter(classA[0, :], classA[1, :], color='b', label='Class A (+1)')
    plt.scatter(classB[0, :], classB[1, :], color='r', label='Class B (-1)')
    plt.title("Generated Linearly-Separable Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Shuffle & combine into X and T
    X2D, T = shuffle_data(classA, classB)  # shape (2, 2n), (2n,)
    X = add_bias_term(X2D)                 # shape (3, 2n)
#--------------------------------------------------------------------------- 
# Part 3.1.2: Compare Perceptron vs Delta Rule (Online and Batch)
#---------------------------------------------------------------------------

# Perceptron (online)
    W_perc, acc_perc = perceptron_training_online(X, T, eta=0.01, epochs=20)
# Delta Rule (online)
    W_delta_online, mse_delta_online = delta_rule_training_online(X, T, eta=0.01, epochs=20)
    # Delta Rule (batch)
    W_delta_batch, mse_delta_batch = delta_rule_training_batch(X, T, eta=0.001, epochs=20)

# Print final results for comparison
    final_acc_perc = compute_accuracy(X, T, W_perc)
    final_acc_delta_online = compute_accuracy(X, T, W_delta_online)
    final_mse_delta_online = mse_delta_online[-1]
    final_mse_delta_batch = mse_delta_batch[-1]

    print(f"Final Accuracy (Perceptron, Online): {final_acc_perc:.2f}")
    print(f"Final Accuracy (Delta, Online): {final_acc_delta_online:.2f}")
    print(f"Final MSE (Delta, Online): {final_mse_delta_online:.4f}")
    print(f"Final MSE (Delta, Batch): {final_mse_delta_batch:.4f}")

#--------------------------------------------------------------------------- 
# Plot final decision boundaries with clear titles
#---------------------------------------------------------------------------

# Perceptron decision boundary
    plot_data_and_boundary(
        X, T, W_perc, 
        title="Perceptron (Online) - Final Decision Boundary\n"
              f"Accuracy: {final_acc_perc:.2f}"
    )

# Delta Rule (Online) decision boundary
    plot_data_and_boundary(
        X, T, W_delta_online, 
        title="Delta Rule (Online) - Final Decision Boundary\n"
            f"Accuracy: {final_acc_delta_online:.2f}, MSE: {final_mse_delta_online:.4f}"
    )

# Delta Rule (Batch) decision boundary
    plot_data_and_boundary(
        X, T, W_delta_batch, 
        title="Delta Rule (Batch) - Final Decision Boundary\n"
            f"MSE: {final_mse_delta_batch:.4f}"
    )

#--------------------------------------------------------------------------- 
# Combined Learning Curves: Perceptron Accuracy and Delta Rule MSE
#---------------------------------------------------------------------------

    plt.figure(figsize=(10, 6))
    plt.plot(acc_perc, label="Perceptron Accuracy (Online)", color='blue', linestyle='-')
    plt.plot(mse_delta_online, label="Delta Rule MSE (Online)", color='red', linestyle='--')
    plt.plot(mse_delta_batch, label="Delta Rule MSE (Batch)", color='green', linestyle='-.')
    plt.title("Learning Curve Comparison: Perceptron vs Delta Rule\n"
            f"Final Accuracy (Perceptron): {final_acc_perc:.2f}, "
            f"Final MSE (Delta Online): {final_mse_delta_online:.4f}, "
              f"Final MSE (Delta Batch): {final_mse_delta_batch:.4f}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy / MSE")
    plt.legend()
    plt.grid(True)
    plt.show()

#--------------------------------------------------------------------------- 
# Initial vs Final Decision Boundaries for Perceptron and Delta Rule
#---------------------------------------------------------------------------

# Initial random weights for Perceptron
    W_initial_perc = 0.01 * np.random.randn(3)
    plot_data_and_boundary(
        X, T, W_initial_perc, 
        title="Perceptron (Online) - Initial Decision Boundary"
    )

# Initial random weights for Delta Rule
    W_initial_delta = 0.01 * np.random.randn(3)
    plot_data_and_boundary(
        X, T, W_initial_delta, 
        title="Delta Rule (Online) - Initial Decision Boundary"
    )

#--------------------------------------------------------------------------- 
# Learning Curves for Perceptron and Delta Rule
#---------------------------------------------------------------------------

# Perceptron accuracy learning curve
    plt.figure()
    plt.plot(acc_perc, label="Perceptron Accuracy")
    plt.title("Perceptron (Online) - Learning Curve\n"
              f"Final Accuracy: {final_acc_perc:.2f}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

# Delta Rule (Online) MSE learning curve
    plt.figure()
    plt.plot(mse_delta_online, label="Delta Rule MSE (Online)")
    plt.title("Delta Rule (Online) - Learning Curve\n"
              f"Final MSE: {final_mse_delta_online:.4f}")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.show()

# Comparison of Delta Rule Online vs Batch
    plt.figure()
    plt.plot(mse_delta_online, label="Delta Rule (Online)")
    plt.plot(mse_delta_batch, label="Delta Rule (Batch)", linestyle="--")
    plt.title("Delta Rule Learning Curve: Online vs Batch\n"
              f"Final MSE (Online): {final_mse_delta_online:.4f}, "
              f"Final MSE (Batch): {final_mse_delta_batch:.4f}")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.show()


    
    #---------------------------------------------------------------------------
    # Part 3.1.2 (3): Remove the bias, re-train with delta rule, batch mode
    #---------------------------------------------------------------------------
    X_no_bias = X2D  # shape (2, 2n)
    W_no_bias, mse_no_bias = delta_rule_training_batch(X_no_bias, T, eta=0.001, epochs=20)
    acc_no_bias = compute_accuracy(X_no_bias, T, W_no_bias)
    print("Final accuracy (Delta, batch, no bias):", acc_no_bias)
    
    # Plot (line forced through origin: W0*x + W1*y = 0 => y=-W0/W1 * x)
    plt.figure()
    classA_nb = X_no_bias[:, T==1]
    classB_nb = X_no_bias[:, T==-1]
    plt.scatter(classA_nb[0,:], classA_nb[1,:], color='b', label='Class +1')
    plt.scatter(classB_nb[0,:], classB_nb[1,:], color='r', label='Class -1')
    x_vals = np.linspace(np.min(X_no_bias[0,:]), np.max(X_no_bias[0,:]), 100)
    if abs(W_no_bias[1]) > 1e-5:
        y_vals = -(W_no_bias[0]/W_no_bias[1]) * x_vals
        plt.plot(x_vals, y_vals, 'k-', label='No-Bias Boundary')
    plt.title("Delta (Batch) without Bias")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    #---------------------------------------------------------------------------
    # Part 3.1.3: Classification of non linearly-separable data
    #---------------------------------------------------------------------------
    mA_ns = [1.0, 0.3]; sigmaA_ns = 0.5
    mB_ns = [0.0, -0.1]; sigmaB_ns = 0.5  # Overlapping distributions
    classA_ns = generate_data_class(mA_ns, sigmaA_ns, n)
    classB_ns = generate_data_class(mB_ns, sigmaB_ns, n)
    X2D_ns, T_ns = shuffle_data(classA_ns, classB_ns)
    X_ns = add_bias_term(X2D_ns)
    
    # Perceptron -> will not converge to perfect separation
    W_perc_ns, acc_perc_ns = perceptron_training_online(X_ns, T_ns, eta=0.01, epochs=20)
    print("Final accuracy on non-separable (Perceptron):", compute_accuracy(X_ns, T_ns, W_perc_ns))
    plot_data_and_boundary(X_ns, T_ns, W_perc_ns, 
                           title="Perceptron on Non-Separable Data (Final Boundary)")
    
    # Delta rule (batch) on the same non-separable data
    W_delta_ns, mse_delta_ns = delta_rule_training_batch(X_ns, T_ns, eta=0.001, epochs=20)
    print("Final accuracy on non-separable (Delta, batch):", compute_accuracy(X_ns, T_ns, W_delta_ns))
    plot_data_and_boundary(X_ns, T_ns, W_delta_ns, 
                           title="Delta on Non-Separable Data (Final Boundary)")
    
    #---------------------------------------------------------------------------
    # Sub-sampling examples: remove 25% or 50% from one class, etc.
    # (For brevity we do just one example below; you can generalize.)
    #---------------------------------------------------------------------------
    # E.g. remove 25% from Class A
    N_total = X2D.shape[1]
    # Indices for classA
    idxA = np.where(T==1)[0]
    idxB = np.where(T==-1)[0]
    n_remove = int(0.25*len(idxA))
    idx_remove = np.random.choice(idxA, size=n_remove, replace=False)
    keep_mask = np.ones(N_total, dtype=bool)
    keep_mask[idx_remove] = False
    X2D_sub = X2D[:, keep_mask]
    T_sub   = T[keep_mask]
    X_sub   = add_bias_term(X2D_sub)
    
    # Train perceptron again
    W_sub, acc_sub = perceptron_training_online(X_sub, T_sub, eta=0.01, epochs=20)
    final_acc_sub = compute_accuracy(X_sub, T_sub, W_sub)
    print("Accuracy after removing 25% of ClassA (Perceptron):", final_acc_sub)
    plot_data_and_boundary(X_sub, T_sub, W_sub, 
                           title="Perceptron after 25% ClassA removed")
    
    # You could do the same with the delta rule, or remove other portions, etc.
    
if __name__ == "__main__":
    main()
