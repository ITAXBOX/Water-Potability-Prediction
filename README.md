# Water Potability Prediction using Machine Learning

![GitHub](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-yellow)

## Table of Contents
1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Dataset Description](#dataset-description)
4. [Objectives](#objectives)
5. [Methodology](#methodology)
   - [Data Preprocessing](#data-preprocessing)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Model Selection and Training](#model-selection-and-training)
   - [Model Evaluation](#model-evaluation)
6. [Results](#results)
7. [Future Work](#future-work)

---

## Overview

Access to clean and safe drinking water is a basic human need, yet millions worldwide do not have access to potable water. Water quality testing is critical to ensuring that water is safe to consume. This project aims to develop a **machine learning model** to predict water potability based on several physicochemical parameters. The goal is to provide a reliable method for quickly monitoring water quality, especially in regions with limited resources for water testing.

---

## Problem Statement

The lack of access to clean drinking water is a global issue, particularly in developing regions. Traditional methods of water quality testing are often time-consuming and resource-intensive. This project addresses the need for a **fast and accurate method** to predict water potability using machine learning. By leveraging physicochemical parameters, the model can classify water samples as potable or non-potable, enabling quicker decision-making and resource allocation.

---

## Dataset Description

The dataset used in this project contains water quality metrics for various water samples. Each sample is labeled as either **potable (1)** or **non-potable (0)**. The dataset includes the following features:

- **pH**: The pH value of the water (a measure of acidity or alkalinity).
- **Hardness**: The amount of dissolved calcium and magnesium in the water.
- **Solids**: Total dissolved solids (TDS) in the water.
- **Chloramines**: The amount of chloramines in the water.
- **Sulfate**: The amount of sulfates dissolved in the water.
- **Conductivity**: The electrical conductivity of the water.
- **Organic_carbon**: The amount of organic carbon in the water.
- **Trihalomethanes**: The amount of trihalomethanes in the water.
- **Turbidity**: The measure of water clarity.
- **Potability**: The target variable indicating whether the water is safe for human consumption (1 = potable, 0 = not potable).

---

## Objectives

1. **Create a supervised machine learning model** to predict water potability.
2. **Assess the model's performance** using accuracy, precision, recall, and the F1 score.
3. **Provide insights** into the primary factors influencing water potability using the dataset.

---

## Methodology

### Data Preprocessing
- **Handling Missing Values:** Missing values in the dataset were imputed using the median value for each feature.
- **Outlier Treatment:** Outliers were capped using the Interquartile Range (IQR) method.
- **Balancing the Dataset:** The dataset was initially imbalanced, with 61% non-potable and 39% potable samples. SMOTE (Synthetic Minority Oversampling Technique) was applied to balance the classes.
- **Feature Scaling:** Features were standardized using `StandardScaler` to ensure uniformity in scale.

### Exploratory Data Analysis (EDA)
- **Distribution Analysis:** Histograms and boxplots were used to visualize the distribution of features.
- **Correlation Analysis:** A correlation heatmap was generated to understand the relationships between features.
- **Target Variable Analysis:** The distribution of the target variable (Potability) was analyzed to identify class imbalance.

### Model Selection and Training
- **Algorithms Used:** K-Nearest Neighbors (KNN) and Decision Trees were selected for their interpretability and ability to handle non-linear relationships.
- **Hyperparameter Tuning:** Grid Search was used to optimize hyperparameters for both models.
- **Training:** The models were trained on 80% of the dataset, with 20% reserved for testing.

### Model Evaluation
- **Metrics:** Accuracy, precision, recall, and F1-score were used to evaluate model performance.
- **Confusion Matrix:** A confusion matrix was generated to visualize the performance of the models.
- **Feature Importance:** For Decision Trees, feature importance was analyzed to understand which features contributed most to the predictions.

---

## Results

### Model Performance

- **KNN:**
  - Accuracy: 67.5%
  - Precision: 69% (Non-Potable), 66% (Potable)
  - Recall: 61% (Non-Potable), 74% (Potable)
  - F1-Score: 65% (Non-Potable), 70% (Potable)

- **Decision Tree:**
  - Accuracy: 62.375%
  - Precision: 62% (Non-Potable), 63% (Potable)
  - Recall: 62% (Non-Potable), 62% (Potable)
  - F1-Score: 62% (Non-Potable), 63% (Potable)

### Key Insights
- The dataset was initially imbalanced, with 61% non-potable and 39% potable samples. SMOTE was applied to balance the classes.
- KNN outperformed Decision Trees, likely due to the low correlation between features, which KNN handles better.
- Feature importance analysis revealed that **pH**, **Hardness**, and **Solids** were the most influential features in predicting water potability.

---

## Future Work

- **Experiment with Advanced Algorithms:** Explore more advanced models like Random Forest, Gradient Boosting, or Neural Networks.
- **Collect More Data:** Gather additional data, especially for the minority class (potable water), to improve model performance.
- **Include Additional Features:** Incorporate features like geographical location, temperature, and seasonal variations to capture more context.
- **Deploy the Model:** Develop a web application or API for real-time water quality monitoring.

---
