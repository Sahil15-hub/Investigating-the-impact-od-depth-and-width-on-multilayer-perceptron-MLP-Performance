# MLP Architecture Exploration — Depth & Width Impact Analysis

This project explores how the **depth** (number of hidden layers) and **width** (neurons per layer) of a **Multilayer Perceptron (MLP)** impact its accuracy, convergence behavior, and ability to generalize.  
Using the classic **Iris dataset**, several neural network architectures are trained and compared to highlight important design trade-offs.

---

## 📘 Project Description

Multilayer Perceptrons are among the earliest neural network models and remain crucial in understanding modern deep learning.  
This study demonstrates:

- How shallow vs. deep architectures behave  
- How narrow vs. wide layers affect model capacity  
- How activation functions (ReLU vs Tanh) alter convergence  
- How architecture influences overfitting and computational cost  

The project includes visualizations such as confusion matrices, learning curves, and PCA plots.

---

## 📊 Dataset: Iris

The Iris dataset contains:

- **150 samples**
- **4 features** (sepal length, sepal width, petal length, petal width)
- **3 species**: *Setosa*, *Versicolor*, *Virginica*

A PCA visualization is used to show class separability, making it ideal for demonstrating model behavior.

---

## 🔧 MLP Architectures Evaluated

| Model Type        | Hidden Layer Structure |
|-------------------|------------------------|
| Shallow Narrow    | (5,)                   |
| Shallow Wide      | (50,)                  |
| Deep Narrow       | (5, 5, 5)              |
| Deep Wide         | (50, 50, 50)           |
| Medium            | (20, 20)               |
| Tanh Variant      | (50,) with tanh        |

**Training Setup:**
- Activation: **ReLU** (unless testing tanh)
- Optimizer: **Adam**
- Max Iterations: **2000**
- Inputs scaled using **StandardScaler**

---

## 📈 Performance Summary

### **Accuracy Results**
- Shallow Narrow: **0.9778**
- Shallow Wide: **1.0000**
- Deep Narrow: **0.9778**
- Deep Wide: **1.0000**
- Medium: **1.0000**
- Tanh Variant: **1.0000**

### Key Takeaways:
- Wider networks converge faster and more accurately on simple datasets like Iris.  
- Depth does not significantly improve performance here and may slow convergence.  
- Medium architectures provide an ideal balance between complexity and accuracy.  
- Activation choice (ReLU vs Tanh) affects learning curves but not final accuracy.

---

## 🧠 Methods & Implementation

The implementation includes:

1. Loading and visualizing the Iris dataset (including PCA projection).  
2. Splitting into **70/30** stratified train–test sets.  
3. Scaling features with **StandardScaler**.  
4. Training models using `MLPClassifier` from scikit-learn.  
5. Generating:
   - Confusion matrices  
   - Learning curves  
   - Classification reports  
   - Accuracy comparisons  
6. Testing activation function variants.

---

## 🧪 Hyperparameter Tuning

Additional experiments included:

- Comparing **ReLU vs Tanh**  
- Using early stopping to prevent overfitting  
- Testing deeper/wider architectures for capacity effects  

Best practices recommended:

- Start simple, then gradually increase network size  
- Use validation curves to detect overfitting  
- Apply cross-validation for reliable evaluation  

---

## 🌍 Ethical Considerations

While the Iris dataset is harmless, real-world applications of MLPs require careful attention to:

- Fairness and bias in datasets  
- Interpretability concerns  
- Environmental impact of training large models  

---

## 📁 Project Structure Example

```
├── data/
├── images/
│   ├── confusion_matrices/
│   ├── learning_curves/
├── mlp_experiments.ipynb
├── mlp_training.py
└── README.md
```

---

## ▶️ How to Run

### Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Run script or notebook
```bash
python mlp_training.py
```
or open `mlp_experiments.ipynb` in Jupyter.

---

## 📚 References

All references used come from the tutorial document provided.

---

