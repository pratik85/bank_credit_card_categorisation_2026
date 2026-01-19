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

## Running the Streamlit App

To launch the interactive web app for card category prediction:

1. Activate your virtual environment (if not already active):
   ```
   env\Scripts\activate  # On Windows
   ```
2. Install Streamlit if not already installed:
   ```
   pip install streamlit
   ```
3. Run the app:
   ```
   streamlit run streamlit_app.py
   ```

This will open a browser window with the app interface for entering customer details and predicting the card category.

## Requirements
See `requirements.txt` for all dependencies.

## Notes
- The dataset should be named `Bank customers.csv` and placed in the project directory.
- The script and notebook will generate model files (`BankCards.pickle`, `BankCardsPCA.pickle`) after training.

## License
This project is for educational purposes.

## GitHub Setup & Usage

To upload and manage this project on GitHub:

1. **Initialize a git repository (if not already done):**
   ```
   git init
   ```
2. **Add all project files:**
   ```
   git add .
   ```
3. **Commit your changes:**
   ```
   git commit -m "Initial commit"
   ```
4. **Create a new repository on GitHub** (via the GitHub website).

5. **Add the remote origin:**
   ```
   git remote add origin https://github.com/your-username/your-repo-name.git
   ```
6. **Push your code to GitHub:**
   ```
   git push -u origin master
   ```

**Steps**

-  git add .
-  git commit -m "Initial IPL Match Predictor project"
-  git branch
-  git status
-  git remote add origin git@github.com:pratik85/bank_credit_card_categorisation_2026
-  git remote -v
-  git push -u origin master

**If you want to do some Changes of some file and update in github**

- git add README.md
- git commit -m "Update README with latest changes"
- git push

