This project focuses on predicting the likelihood of diabetes using machine learning techniques applied to patient health records and diagnostic measurements. The project is inspired by the research paper "Diabetes Prediction and Management Using Machine Learning Approaches" (2025) and incorporates concepts from feature selection research to improve predictive performance. Early identification of diabetes can help healthcare providers take preventive actions and support timely treatment decisions. The objective is to build and compare multiple machine learning models capable of classifying whether a patient is likely to have diabetes based on clinical and demographic information.

Number of samples: 768 patients
Number of features: 8 clinical attributes

Common features include:
  Pregnancies
  Glucose Level
  Blood Pressure
  Skin Thickness
  Insulin
  Body Mass Index (BMI)
  Diabetes Pedigree Function
  Age

Target variable:
  0 → No Diabetes
  1 → Diabetes

Several machine learning algorithms were trained and evaluated, including:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Random Forest
5. Gradient Boosting

To ensure reliable model assessment, the following evaluation methods were used:
  Train/Test Split
  5-Fold Cross Validation
  Accuracy
  Precision
  Recall
  F1 Score
  ROC Curve
  AUC-ROC

https://arxiv.org/pdf/2506.11501

Conclusion:
This project demonstrates how machine learning techniques can be used to predict diabetes from patient health data. By comparing multiple classification algorithms and incorporating feature selection methods, the project highlights practical approaches for improving predictive accuracy and supporting data-driven healthcare decision-making.
