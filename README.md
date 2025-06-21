# Single-Layer Perceptron & Learning Rules

## Project Overview
This project explores **single-layer perceptrons** and their learning rules, implementing both the **Perceptron Learning Algorithm** and the **Widrow-Hoff (Delta Rule)** for classification tasks. The objective is to understand their limitations, performance under varying learning rates, and ability to classify **linearly separable vs. non-linearly separable data**.

## Technologies & Tools Used
- **Programming Language**: Python
- **Libraries**: NumPy, Matplotlib, Scikit-Learn
- **Learning Rules**: Perceptron Rule, Widrow-Hoff Delta Rule
- **Training Approaches**: Sequential vs. Batch Learning

## Key Features
- **Decision Boundaries Analysis**: Visualizing how perceptrons learn classification tasks.
- **Error Rate Evaluation**: Measuring performance under different learning rates.
- **Batch vs. Sequential Learning**: Comparing stability and convergence speed.
- **Effect of Bias**: Understanding how decision boundaries change when bias is removed.

## Results
- **Perceptron Learning Algorithm** converges successfully for linearly separable data.  
- **Delta Rule** provides smoother decision boundaries but struggles with large learning rates.  
- **Batch learning** results in more stable training but is slower than **sequential updates**.  
- **Non-linearly separable data** proves to be a major limitation, confirming the need for multi-layer networks.  
