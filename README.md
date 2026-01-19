# Self Bank Customer Analysis

This project analyzes bank customer data to predict card categories using machine learning techniques. It includes data preprocessing, exploratory data analysis, dimensionality reduction, and model training with Random Forest.

## Project Structure
- `Self_Bank.ipynb`: Jupyter notebook with step-by-step analysis and modeling.
- `Self_Bank.py`: Python script version of the notebook.
- `Bank customers.csv`: Dataset containing customer information.
- `requirements.txt`: List of required Python packages.

## Setup Instructions
1. **Create a virtual environment** (recommended):
   ```
   python -m venv env
   env\Scripts\activate  # On Windows
   ```
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Run the notebook:**
   Open `Self_Bank.ipynb` in Jupyter or VS Code and run the cells step by step.

4. **Run the script:**
   ```
   python Self_Bank.py
   ```

## Main Features
- Data cleaning and exploration
- Visualization of customer age and credit utilization
- Encoding of categorical variables
- Dimensionality reduction using PCA
- Random Forest classification for card category prediction
- Model saving with pickle

## Requirements
See `requirements.txt` for all dependencies.

## Notes
- The dataset should be named `Bank customers.csv` and placed in the project directory.
- The script and notebook will generate model files (`BankCards.pickle`, `BankCardsPCA.pickle`) after training.

## License
This project is for educational purposes.
