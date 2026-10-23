# Credit-Card-Fraud-Detection
# 💳 Credit Card Fraud Detection

A Machine Learning web application that detects potentially fraudulent credit card transactions using **Random Forest Classifier**, **Scikit-learn**, and **Streamlit**.

## 📌 Project Overview

Credit card fraud is a major challenge in digital payments. This project uses machine learning to classify transactions as legitimate or fraudulent based on transaction features.

The application provides an interactive Streamlit interface where users can enter transaction details and receive a prediction from the trained Random Forest model.

## 🎯 Objectives

- Detect potentially fraudulent credit card transactions.
- Apply machine learning classification using Random Forest.
- Understand and handle imbalanced datasets.
- Evaluate model performance using appropriate classification metrics.
- Build an interactive web application using Streamlit.

## 🛠️ Technologies Used

- **Python** – Programming language
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computing
- **Scikit-learn** – Model training and evaluation
- **Random Forest Classifier** – Machine learning algorithm
- **Streamlit** – Web application interface
- **Matplotlib / Seaborn** – Data visualization (if used)

## 📂 Dataset

**Dataset:** Credit Card Fraud Detection

**Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The original dataset contains 284,807 transactions, including 492 fraudulent transactions.

The target column is `Class`:

- `0` – Legitimate transaction
- `1` – Fraudulent transaction

Most input features are anonymized using Principal Component Analysis (PCA). The dataset also contains `Time` and `Amount`.

## ⚙️ Machine Learning Workflow

1. Load and explore the dataset.
2. Perform data preprocessing and prepare the input features.
3. Split the data into training and testing sets.
4. Train the Random Forest Classifier using Scikit-learn.
5. Evaluate the model using suitable classification metrics.
6. Save the trained model for reuse.
7. Build a Streamlit application for interactive predictions.

## 🌲 Model Used: Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees to produce a final classification.

**Why Random Forest?**

- Captures complex relationships between features.
- Handles nonlinear classification problems.
- Combines multiple trees to improve predictive performance.
- Provides feature importance estimates.
- Supports binary classification.

## 📊 Model Evaluation

The model can be evaluated using:

- **Precision** – Proportion of predicted fraud cases that are actually fraudulent.
- **Recall** – Proportion of actual fraud cases detected by the model.
- **F1-score** – Harmonic mean of precision and recall.
- **Confusion Matrix** – Shows correct and incorrect classifications.
- **Precision-Recall AUC** – Useful for evaluating performance on imbalanced datasets.

Accuracy alone is not sufficient because fraudulent transactions represent a very small proportion of the dataset.

*Add your actual evaluation scores here after testing your trained model.*

## 🖥️ Streamlit Web Application

The Streamlit interface allows users to interact with the trained model and obtain transaction predictions.

**Application features may include:**

- User-friendly transaction input form.
- Fraud or legitimate transaction prediction.
- Display of prediction results.
- Fraud probability display, if implemented.
- Interactive visualizations, if implemented.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Alternatively, install the main libraries:

```bash
pip install pandas numpy scikit-learn streamlit
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

**Note:** Ensure that the dataset and saved model files are available at the paths expected by your code.

## 📁 Project Structure

```text
credit-card-fraud-detection/
│
├── app.py                 # Streamlit application
├── model.pkl              # Trained Random Forest model (if used)
├── creditcard.csv         # Dataset (if included locally)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

Update the filenames above to match your actual repository structure.

## 🔮 Future Improvements

- Compare Random Forest with other classification algorithms.
- Tune model hyperparameters for improved performance.
- Optimize the classification threshold.
- Apply suitable techniques to handle class imbalance.
- Deploy the application online.
- Improve the interface with additional performance visualizations.

## 👨‍💻 Author

Aryan Rawat



⭐ If you find this project useful, consider giving the repository a star.

**Disclaimer:** This project is for educational purposes and is not a production-ready financial fraud detection system.
