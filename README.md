## Installation of Intel OneAPI Toolkit

To optimize our project using Intel hardware, we installed the Intel oneAPI AI Tools as follows:

**Downloaded Intel AI Tools**:
   We downloaded and installed Intel AI Tools for full hardware optimization.
   ```bash
   wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/f27e9e0e-ec27-4024-a4bf-b30c48c99564/l_AITools.2024.2.0.156.sh
   sh l_AITools.2024.2.0.156.sh
   ```

**Set Up Conda Environments**:
   We created isolated environments for different phases of the project:
   ```bash
   conda create -n modin intelpython3_core
   conda activate modin
   ```

   ```bash
   conda create -n pytorch-gpu intelpython3_core
   conda activate pytorch-gpu
   ```

**Installed Dependencies**:
   Using a `requirements.txt` file, we ensured that all necessary dependencies were installed:
   ```bash
   pip install -r requirements.txt
   ```
## Data Preprocessing with Modin

We used **Modin**, optimized by Intel, for data preprocessing. This enabled us to parallelize and speed up data operations, which was essential given the large size of the transportation datasets we worked with.

**Activated the Modin Environment**:
   ```bash
   conda activate modin
   ```

**Ran Preprocessing Script**:
   We executed the preprocessing script, which cleaned and organized the data for modeling:
   ```bash
   python preprocess_data.py
   ```

   Using Modin, we parallelized common `pandas` operations to handle the dataset efficiently, leveraging Intel’s optimization.

---

## Feature Extraction

We performed feature extraction by selecting key attributes from vehicle data and generating new features relevant to public transportation. These features were instrumental in improving the accuracy of our models.

To perform feature extraction and generate new features, simply run the following command:

```bash
python feature_extraction.py
```

---

## Exploratory Data Analysis (EDA)

Using **Modin** and visualization tools like **Seaborn**, we conducted extensive exploratory data analysis. This helped us identify patterns, correlations, and trends within the data, allowing us to refine our model inputs and hypotheses.

```bash
python eda.py
```

---

## Model Training with PyTorch

For model training, we utilized **PyTorch**, with Intel GPU support, to accelerate the process. By distributing computation tasks across GPUs, we significantly reduced training time.

**Activated PyTorch-GPU Environment**:
   ```bash
   conda activate pytorch-gpu
   ```

**Trained the Model**:
   We ran our training script, which loaded the preprocessed data and trained the deep learning model:
   ```bash
   python train_model.py
   ```

   This phase involves testing various architectures and optimizing performance using Intel’s PyTorch extensions for better resource utilization.

---

## Model Optimization with BOA and Orion

To optimize the model, we employed the **Butterfly Optimization Algorithm (BOA)** and **Orion**. These tools helped us fine-tune hyperparameters and improve the overall model performance.

- **BOA** allowed us to perform global optimization across various hyperparameters.
- **Orion** facilitated automated hyperparameter tuning for enhanced model accuracy and efficiency.

By leveraging these optimization techniques, we were able to increase model efficiency and performance, ensuring better predictions for transportation schedules and routes.

---

## Deployment with Flask

Once our model was trained and optimized, we deployed it using **Flask**.

1. **Activated the PyTorch-GPU Environment**:
   ```bash
   conda activate pytorch-gpu
   ```

2. **Deployed the Model with Flask**:
   Flask allowed us to deploy the model as a web service, enabling real-time predictions for public transportation schedules:
   Flask, combined with Intel optimizations, ensured that the deployment was smooth and efficient, allowing us to serve predictions at scale.

---

## Prototype Details

Our AI-based system integrated large-scale transportation data with deep learning models to predict optimized schedules and routes. The system dynamically adjusts to demand patterns, reducing fuel consumption and emissions, while maintaining user convenience.

---

## Utility Summary

Through this project, we achieved:

- Optimized transportation routes and schedules
- Reduced fuel consumption and minimized carbon emissions
- Real-time predictions using an AI model deployed via Flask
- Efficient use of Intel-optimized tools to accelerate model training and deployment

The project demonstrates how AI can be applied to public transportation to enhance environmental sustainability and operational efficiency.
