# Public Transportation Efficiency Project

This project utilizes large datasets to improve public transportation efficiency through optimized deep learning models and advanced optimization algorithms. We leverage Intel-optimized resources like **Modin**, **PyTorch**, and **Flash** for efficient data handling and model deployment. Additionally, root optimization algorithms such as the **Butterfly Optimization Algorithm (BOA)** and the **Orion package** are used to fine-tune the model.

## Prerequisites

1. **Intel oneAPI AI Tools**: Install for Intel-specific optimizations.
2. **Conda**: Create virtual environments for isolated package management.
3. **VSCode with WSL**: Supports GPU training for deep learning tasks.

### Installation

1. **Download Intel AI Tools**

   ```bash
   wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/f27e9e0e-ec27-4024-a4bf-b30c48c99564/l_AITools.2024.2.0.156.sh
   sh l_AITools.2024.2.0.156.sh
   ```

   Accept the license terms and install the Intel AI Tools.

2. **Create Conda Environments**

   Set up specific environments for different stages:
   
   ```bash
   conda create -n modin intelpython3_core
   conda activate modin
   ```
   
   ```bash
   conda create -n pytorch-gpu intelpython3_core
   conda activate pytorch-gpu
   ```

3. **Install the required packages**

   Use the `requirements.txt` file to install dependencies for the project:

   ```bash
   pip install -r requirements.txt
   ```

### requirements.txt

Here’s the list of dependencies required for this project:

```
modin[ray]
torch
torchvision
intel-oneapi-modin
intel-oneapi-pytorch
flash
scikit-learn
xgboost
butterfly-optimization-algorithm
orion
```

Make sure to install **Orion** and **Butterfly Optimization Algorithm (BOA)** for model optimization tasks.

### 1. Data Preprocessing with Modin

Modin (optimized by Intel) allows parallel processing of large datasets. This helps in speeding up preprocessing for massive data.

**Steps:**

1. **Activate the Modin Environment:**

   ```bash
   conda activate modin
   ```

2. **Run Preprocessing:**

   Execute the preprocessing script:

   ```bash
   python preprocess_data.py
   ```

   **Modin for Data Processing:**
   Modin automatically parallelizes `pandas` operations, utilizing Intel-optimized hardware to efficiently process large datasets.

### 2. Model Training with PyTorch

We leverage **PyTorch**, which uses GPU acceleration for deep learning model training.

**Steps:**

1. **Activate PyTorch-GPU Environment:**

   ```bash
   conda activate pytorch-gpu
   ```

2. **Run the Training Script:**

   ```bash
   python train_model.py
   ```

   The model training script performs:
   - Loading the preprocessed data
   - Training the deep learning model
   - Utilizing Intel GPU optimizations to reduce training time

### 3. Model Optimization with BOA and Orion

We use root optimization algorithms like **Butterfly Optimization Algorithm (BOA)** and **Orion** to improve the performance of our model.

**Using BOA**:

In your training script, implement BOA as follows:

```python
from butterfly_optimization_algorithm import BOA

# Define the optimization problem
def objective_function(params):
    # Example: Optimizing model's hyperparameters
    return model.evaluate(params)

# Initialize BOA
optimizer = BOA(objective_function, num_iterations=100, population_size=50)
best_params = optimizer.optimize()
```

**Using Orion**:

Orion can be used to manage experiments for tuning hyperparameters:

```bash
pip install orion
orion hunt --config orion_config.yaml python train_model.py
```

Orion will automate hyperparameter optimization for the model.

### 4. Model Analysis with PyTorch

Once the model is trained, evaluate its performance using PyTorch’s built-in tools:

```python
model.eval()
test_loss, accuracy = model.evaluate(test_data, test_labels)
```

### 5. Deployment with Flash

Once the model is optimized and trained, use **Flash** for deployment:

1. **Activate the PyTorch-GPU Environment:**

   ```bash
   conda activate pytorch-gpu
   ```

2. **Deploy Using Flash:**

   Flash allows easy deployment of models:

   ```python
   from flash import serve

   serve(model, host="0.0.0.0", port=8000)
   ```

**Intel Flash Optimizations:**
Flash works efficiently with Intel hardware, optimizing GPU performance during deployment.

## Conclusion

In this project, we combined Intel’s hardware optimizations with advanced tools like **Modin** for data pre-processing, **PyTorch** for model training, and **Flash** for deployment. Root optimization algorithms like the **Butterfly Optimization Algorithm (BOA)** and **Orion** further enhance model performance by tuning key parameters.
```