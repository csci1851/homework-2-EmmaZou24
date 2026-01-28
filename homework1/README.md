# Homework 1: Introduction to Regression and Classification

## Overview
In this homework, you will work with two datasets to explore foundational models in machine learning: linear regression and classification. 

You will:

- Load and inspect biomedical datasets.
- Perform classification (heart disease).
- Use classification models to learn important features.
- Perform regression (biological aging prediction).
- Learn foundations of feature selection and model evaluation.  

---

## Datasets

You will work with two datasets:
1. Heart Disease Dataset (Classification)
- **Goal**: Predict presence/absence of heart disease.
- **Features**: Demographics, cholesterol, ECG results, etc.

2. Biological Aging Dataset (Regression)
- **Goal**: Predict biological age from DNA methylation patterns.
- **Source**: Public high-dimensional DNA methylation dataset (GSE139307).
- **Features**: Preprocessed and normalized DNA methylation features (epigenetic markers), with samples grouped by tissue type (blood WBC, blood leukocyte, saliva, kidney).

---

## Installation

Install dependencies using pip:

1. **Clone** this repo:
   ```bash
   git clone git@github.com:brown-csci1851/stencil.git
   cd stencil/homework1
   ```
2. Create virtual environment:
    ```bash
    python -m venv .hw1
    ```
3. Install dependencies:
    ```bash
    source .hw1/bin/activate (Linux/MacOS) or .\.hw1\Scripts\activate (Windows-PowerShell)
    pip install -r requirements.txt
    ```

After creating and activating the virtual environment, select it as the Jupyter kernel in `src/playground.ipynb` to run the notebook using the same installed dependencies.

---
## Data Normalization and Missing Values

Both datasets are already preprocessed, but they differ in how scaling should be handled during modeling.

For the heart disease dataset, features are measured on different scales (e.g., age, cholesterol, blood pressure). Feature scaling can therefore affect model performance. You are encouraged to compare models trained **with and without feature scaling** (e.g., standardization) and analyze the impact on evaluation metrics.

For the biological aging (DNA methylation) dataset, features are normalized methylation beta values between 0 and 1. No additional biological normalization is needed. You may still apply feature scaling (e.g., z-score standardization) before regression so that features are on similar scales, especially when using regularized models.

Both datasets may contain missing values. You should identify and handle missing data appropriately (e.g., using feature-wise mean or median imputation), and ensure that any imputation or scaling is fit only on the training data to avoid data leakage.

---

## Tasks

You will complete the TODOs in `src/model.py` and `src/playground.ipynb` to accomplish the following tasks:

- [ ] Load both datasets using your HW1DataLoader.  (The code is provided for you!)

### Heart Disease Dataset
- [ ] Visualize distributions of numeric features, class balance (heart disease), outliers and missing values.
- [ ] Train a logistic regression model on the heart disease dataset with and without feature transformation.
- [ ] Evaluate with accuracy, precision, recall, F1-score using K-fold cross-validation.

### Biological Aging Dataset
- [ ] Train a linear regression model on the biological aging (DNA methylation) dataset, split by tissue type.
- [ ] Evaluate models using MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and \(R^2\).
- [ ] Evaluate the generalizability of tissue-specific models to other tissue types.
- [ ] Train additional models using limited feature selection (e.g., selecting the most variable methylation features) and evaluate their performance.
- [ ] Visualize predicted vs. actual age using scatter plots.

---

## Common Machine Learning Practices

Throughout this assignment, make sure to follow these standard machine learning best practices:

* Never train your model on the full dataset — always split the data into training and validation sets.
* Always examine summary statistics of your data (for regression tasks, we want feature values to be similarly scaled).
* Always check for missing values and handle them appropriately.

---

## Final Reflection

You will then write a **2-3 page reflection** that includes **figures** and **interpretation** of your results. Your write-up should clearly reference the plots and metrics you generated (not just final numbers).

Answer the following questions (with figures/screenshots where relevant):
* What are the main characteristics of the heart disease dataset and its features?
* What do the feature distributions look like, how balanced are the classes (heart disease present/absent), and are there any outliers or missing values?
* What does your confusion matrix show, and what does your ROC curve indicate about performance?
* What were your cross-validation results (summarize in a table or plot)?
* What were your accuracy, precision, recall, F1-score, and ROC-AUC results?
* Which features were most important and least important for classification, and why?
* What are the main limitations of the dataset and the logistic regression model?
* How did feature scaling or transformations change your results (ROC-AUC and/or decision boundary)?

* What are the main characteristics of the biological aging (DNA methylation) dataset and the specific tissues present in the dataset?
* How well did tissue-specific regression models perform (MAE, RMSE, \(R^2\))?
* How well did models generalize across tissues? Summarize results in a heatmap or table showing train tissue \(\rightarrow\) test tissue.
* How closely did predicted ages match true ages? Include predicted vs. actual scatter plots (e.g. one per tissue).
* How did performance change when using limited feature sets (e.g., top-variance methylation features) compared to full-feature models?
* What other feature selection strategies could be applied beyond selecting top-variance features?
* What are the main limitations of the dataset and linear regression models (data quality, modeling assumptions, generalizability)?
* Which tissues generalized poorly, and why might that be?
---

## Expected Skills

By the end of this homework, you should be able to:

* Perform classification and regression using linear and logistic models.
* Explore feature selection and model evaluation in biological contexts.
* Visualize and interpret simple model predictions.

