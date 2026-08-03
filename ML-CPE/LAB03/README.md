Executive Summary of Facial Analysis and Predictive Modeling

Project Objectives and Scope
This technical lab evaluated an end-to-end Machine Learning pipeline applied to the UTKFace Dataset, covering Regression, Classification, and Overfitting Analysis[cite: 2]. The primary objective was to benchmark model performances across two core tasks: continuous Age Prediction (Regression) and discrete Gender Classification (Classification), utilizing feature reduction via Principal Component Analysis (PCA), decision boundary visualization, and cross-metric verification to evaluate model generalizability[cite: 2].

---

Key Methodologies and Implementation
LAB 1: Regression Performance (Age Prediction)
   
Multiple regression architectures were evaluated to predict continuous human age values from facial features[cite: 2]:
-Simple Linear Regression:** Served as a baseline model, achieving a Mean Absolute Error (MAE) of 14.99 years and a coefficient of determination (R²) of 0.0201.
-Multiple Linear Regression:** Improved predictive performance significantly by utilizing multi-feature inputs, yielding an MAE of 12.86 years and R² of 0.2778.
-Random Forest Regressor:** Delivered the optimal regression performance, minimizing the prediction error to an MAE of 11.95 years and achieving an R² of 0.3446.
-Performance Visualization:** Model accuracy was evaluated using scatter plots comparing predicted vs. actual age against the ideal reference line (y = x).
LAB 2: Classification Metrics & Decision Boundaries (Gender Prediction)
   
Binary classification techniques were implemented to predict gender (Male vs. Female) through statistical metrics and spatial projection[cite: 2]:
-Performance Metrics:** The model achieved an overall Accuracy of 77.10%, Precision of 78.20%, Recall of 73.46%, and an F1-Score of 0.7576[cite: 2].
-Confusion Matrix Evaluation:** Evaluation across 4,572 test instances yielded 1,889 True Males, 1,636 True Females, 591 False Males, and 456 False Females.
-Dimensionality Reduction & Decision Boundary:** Principal Component Analysis (PCA) was used to project high-dimensional features into a 2D space (Component 1 vs. Component 2) to visualize the decision boundary separating Male and Female classes[cite: 2].
LAB 3: Model Comparison & Overfitting Evaluation
   
A comparative generalization analysis was performed to measure potential overfitting and model stability[cite: 2]:
-Training vs. Testing Performance:** The classification model displayed stable learning with a Training Accuracy of 77.84% versus a Testing Accuracy of 77.10%, resulting in a minimal Performance Gap of 0.74% (indicating high generalization without significant overfitting)[cite: 2].
-Discriminative Capacity:** Receiver Operating Characteristic (ROC) curve analysis demonstrated strong class separation capability, achieving an Area Under the Curve (AUC) of 0.8538[cite: 2].

---

Data Source and Reference

All data utilized throughout this modeling and evaluation workflow was retrieved from the following verified repository:

Dataset Title: UTKFace Dataset (Large-Scale Face Dataset)
Author/Publisher: Chirag Saipanuganti
Platform: Kaggle
Official URL: https://www.kaggle.com/datasets/chiragsaipanuganti/utkface
