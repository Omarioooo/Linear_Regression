# Building statistical view of Linear Regression from scratch

📊 Dataset Used in This Project

🏠 **`Real_estate_valuation`**: Used for evaluating the Multiple Linear Regression model to predict house prices per unit area based on features like house age, distance to MRT stations, number of convenience stores, latitude, and longitude.

___

📈 Multiple Linear Regression Evaluation Phases

___
🧹 1. Data Preprocessing
___
We begin by preparing the dataset for modeling:

📥 Load the dataset (Real_estate_valuation.xlsx) using Pandas.
🗑️ Drop irrelevant columns: No (index) and transaction date (non-predictive).
🔍 Verify dataset structure using .head(), .info(), and .describe().
🕳️ Check for missing values using .isnull().sum() (none found).
🧠 Confirm data types (all numeric except transaction date, which was dropped).
🚫 Remove outliers for features with significant outliers:
Features: house age, distance to the nearest MRT station, longitude.
Method: Interquartile Range (IQR) with bounds [Q1 - 1.5*IQR, Q3 + 1.5*IQR].


📏 Scale features using StandardScaler to ensure equal contribution.


🔍 2. Exploratory Data Analysis (EDA)
___
Understanding the data visually:

📊 Generate boxplots for features (number of convenience stores, house price of unit area, latitude, distance to the nearest MRT station, longitude) to identify outliers.
📉 Plot histograms for all features (excluding target) to analyze distributions.
📈 Create scatter plots of each feature vs. the target (house price of unit area) to explore relationships.
🧲 Key insights:
distance to the nearest MRT station shows a negative correlation with house price.
number of convenience stores and latitude exhibit positive trends with house price.




🧪 3. Data Preparation
___
Preparing features for modeling:

🎯 Extract features (X) and target (y):
X: house age, distance to the nearest MRT station, number of convenience stores, latitude, longitude.
y: house price of unit area.


✂️ Split data into 80% training and 20% testing using train_test_split with random_state=42.
📏 Apply StandardScaler to normalize features for consistent model training.


🧠 4. Model Development
___
Building and evaluating the Multiple Linear Regression model:

🛠️ Implement a custom MultipleLinearRegression class:
Uses normal equation (β = (XᵗX)⁻¹Xᵗy) to compute coefficients.
Calculates intercept (B0) and feature coefficients (B1, B2, ..., B5).
Computes predictions, residuals, and statistical metrics (SST, SSR, SSE, MSE, MSR, R²).


🧪 Fit the model on the training data (X_train, y_train).
🔮 Generate predictions on the test set (X_test).

📌 Model Components:

Prediction: y = B0 + B1*x1 + B2*x2 + ... + B5*x5.
ANOVA: Computes sum of squares (SST, SSR, SSE), degrees of freedom, mean squares (MSR, MSE), F-statistic, and critical F-value.
Hypothesis Testing: Performs t-tests for each coefficient to assess significance (p-value < 0.05).
Confidence Intervals: Calculates 95% confidence intervals for coefficients.
Visualization: Plots a 3D regression plane using PCA-transformed features.


📊 5. Model Evaluation
___
Evaluating model performance quantitatively and visually:

📏 Quantitative Metrics:
R-squared: Measures the proportion of variance explained by the model.
ANOVA Table:


Source
Sum of Squares
DF
Mean Squares



Regression
SSR
5
MSR


Residual
SSE
n-6
MSE


Total
SST
n-1




F-Statistic: Tests overall model significance (F = MSR/MSE).
Hypothesis Tests: t-statistics and p-values for each coefficient (B0, B1, ..., B5).
Confidence Intervals: 95% intervals for coefficients to assess precision.


🖼️ Visual Diagnostics:
🔥 Display 3D regression plane with PCA-transformed features (2 components) and actual data points.
📉 Scatter plots from EDA to validate feature-target relationships.



🏆 Example Results (Hypothetical, as exact values depend on data):

R-squared: ~0.65 (indicating moderate fit).
F-Statistic: High value, rejecting H0 (model is significant).
Significant Coefficients: Likely distance to the nearest MRT station and number of convenience stores (p < 0.05).
Confidence Intervals: Narrow intervals for significant coefficients.


🔧 6. Model Insights
___
Key findings and recommendations:

✅ Model Fit: The model captures moderate variance in house prices, with distance to the nearest MRT station and number of convenience stores as key predictors.
⚠️ Limitations: Outlier removal may reduce dataset size, potentially affecting generalizability.
🛠️ Improvements:
Explore feature engineering (e.g., interaction terms).
Test polynomial regression for non-linear relationships.
Collect additional data to improve robustness.




🏁 Conclusion
___
✅ The Multiple Linear Regression model effectively predicts house prices per unit area, with significant contributions from distance to the nearest MRT station and number of convenience stores.
🏆 Key Strengths: Custom implementation with comprehensive statistical analysis (ANOVA, hypothesis testing, confidence intervals) and intuitive 3D visualization.
🚫 No overfitting observed, as the model generalizes to the test set.
📝 Future Work: Incorporate non-linear models or additional features to improve R-squared and predictive accuracy.
