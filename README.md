# Connecting Intel OneAPI Toolkit GPU to VSCode for Jupyter Notebooks

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Setting Up VSCode](#setting-up-vscode)
4. [Running Jupyter Notebooks](#running-jupyter-notebooks)
5. [Performing Data Analysis](#performing-data-analysis)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

Before you start, ensure you have the following:

- A system with an Intel GPU (e.g., Intel Iris Xe).
- Intel OneAPI Toolkit installed on your machine.
- Visual Studio Code (VSCode) installed.
- Python installed (preferably in a virtual environment).

## Installation Steps

1. **Install Intel Graphics Drivers:**
   - Download and install the latest Intel graphics drivers from the [Intel Download Center](https://downloadcenter.intel.com/).

2. **Download Intel OneAPI Toolkit:**
   - Go to the [Intel OneAPI toolkit page](https://software.intel.com/content/www/us/en/develop/tools/oneapi.html) and download the toolkit.

3. **Install Intel OneAPI Toolkit:**
   - Follow the installation instructions provided on the Intel site. Choose the components related to Deep Learning.

4. **Set Up Environment Variables:**
   - Open a terminal and run the setup script to set environment variables:
     ```bash
     source /opt/intel/oneapi/setvars.sh
     ```

5. **Install Required Python Libraries:**
   - Ensure you have the necessary libraries installed in your Python environment:
     ```bash
     pip install numpy pandas scikit-learn matplotlib seaborn torch torchvision torchaudio intel-tensorflow
     ```

## Setting Up VSCode

1. **Open VSCode:**
   - Launch Visual Studio Code.

2. **Install Python Extension:**
   - Go to the Extensions view (`Ctrl+Shift+X`) and search for "Python" to install the Microsoft Python extension.

3. **Install Jupyter Extension:**
   - Similarly, search for "Jupyter" in the Extensions view and install the Jupyter extension by Microsoft.

4. **Create or Open Your Jupyter Notebook:**
   - You can create a new Jupyter Notebook by creating a `.ipynb` file or open an existing one.

5. **Select Python Interpreter:**
   - Click on the Python interpreter in the bottom-left corner of VSCode and select the interpreter associated with your virtual environment where the Intel libraries are installed.

## Running Jupyter Notebooks

1. **Open Your Jupyter Notebook (.ipynb):**
   - Navigate to your `.ipynb` file and open it.

2. **Connect to the Intel GPU:**
   - Use the following code snippet in a code cell to verify if the Intel GPU is available:
   ```python
   import torch

   print("Is CUDA available:", torch.cuda.is_available())
   print("Available devices:", torch.cuda.device_count())
   print("Current device:", torch.cuda.current_device())
