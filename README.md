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

## Running the Project
Follow the steps below in the specified order to run the project:

1. **Preprocess the Data**
   - **File:** `preprocess_data.py`
   - **Description:** This script cleans and preprocesses the dataset. Ensure your data files are correctly placed and adjust file paths as needed.
   - **Command:** 
     ```bash
     python preprocess_data.py
     ```

2. **Feature Extraction**
   - **File:** `feature_extraction.py`
   - **Description:** This script extracts relevant features from the cleaned data for model training.
   - **Command:** 
     ```bash
     python feature_extraction.py
     ```

3. **Train the Model**
   - **File:** `orion_butterfly.py`
   - **Description:** This script trains the AI model using the preprocessed features and saves the trained model for future predictions.
   - **Command:** 
     ```bash
     python orion_butterfly.py
     ```

4. **Make Predictions**
   - **File:** `train_model.py`
   - **Description:** This script loads the trained model and contains the function to make predictions based on new input data.
   - **Command:** This file is indirectly run via the Streamlit application in the next step.

5. **Run the Streamlit Application**
   - **File:** `app.py`
   - **Description:** This is the main application where users can input data for predictions and visualize results through interactive graphs.
   - **Command:** 
     ```bash
     streamlit run app.py
     ```

6. **Exploratory Data Analysis**
   - **File:** `eda.py`
   - **Description:** This script generates various visualizations using Plotly based on the dataset. The visualizations will be displayed in the Streamlit application when running `app.py`.

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
