# Indian Liver Disease Prediction

## Project Overview
Machine learning classification project using the Indian Liver Patient Dataset (ILPD) to predict liver disease.

## Objective
To compare classification algorithms and evaluate predictive performance.

## Dataset
Indian Liver Patient Dataset (ILPD)

## Methods Used
- Data cleaning
- Feature engineering
- Logistic Regression
- Random Forest
- Model evaluation (Accuracy, Precision, Recall, F1-score)

## Results
Best performing model:  Random Forest


F1-Score: 0.847
Recall: 0.904
Accuracy: 0.769
Total Cost: 59.0

1. Cost-sensitive learning significantly improves recall for liver disease detection.
2. Optimal cost ratio is 5:1 (FN:FP), balancing sensitivity and specificity.
3. XGBoost performs best among all algorithms tested.
4. Gender-based fairness analysis reveals performance disparities.
5. Feature importance aligns with clinical knowledge (bilirubin, albumin key).
 

## Tools
Python, R and Jamovi
