Retirement Readiness Predictor
A complete machine-learning portfolio project that predicts a customer's Expected_Retirement_Fund from demographic, income, savings, balance-sheet, and behavioural attributes.

The project follows the ML workflow from exploratory data analysis to model development, evaluation, explainability, business interpretation, and a reusable prediction pipeline. The final outputs are prepared for a Power BI dashboard.

Important: The dataset is synthetic. Results demonstrate the project workflow on generated data and should not be interpreted as evidence about real retirement savers or used for real financial advice.

Project status
All seven notebooks are complete and uploaded to GitHub. The Power BI dashboard is the remaining project stage.

Stage

Status

Purpose

01 — Exploratory Data Analysis

Complete

Understand the dataset, investigate data quality and relationships, and identify target leakage.

02 — Data Preprocessing & Feature Engineering

Complete

Build a reusable pipeline to clean, transform, and prepare model inputs.

03 — Baseline Model Training & Evaluation

Complete

Compare baseline regression models and establish a performance benchmark.

04 — Model Improvement

Complete

Tune Gradient Boosting, compare modeling choices, and select the final model.

05 — Model Evaluation & Explainability

Complete

Evaluate the frozen model, examine prediction errors, and explain model behaviour.

06 — Final Business Report

Complete

Translate model predictions into business measures and customer-readiness categories.

07 — Deployment & Prediction Pipeline

Complete

Apply the saved model to new synthetic customers and export Power BI-ready files.

Power BI Dashboard

In progress

Build an interactive business dashboard using the exported predictions and supporting results.

1. The prediction problem
Target
Expected_Retirement_Fund

This is a continuous value representing a customer's expected retirement fund.

Machine-learning task
Supervised regression

The model uses customer attributes to predict the expected retirement fund. Because the target is strongly right-skewed, the main model is trained in log space and its predictions are converted back to dollars with a smearing correction.

The model predicts the fund amount. It does not directly predict a customer's retirement-readiness category.

Business interpretation
After the prediction, the project calculates measures such as:

Funding Gap — retirement goal minus predicted retirement fund.

Funding Ratio — predicted retirement fund divided by retirement goal.

Retirement Readiness — a business-rule category based on the funding ratio.

This separates the machine-learning prediction from the business rules used to interpret it.

2. Dataset and preparation
The project uses 30,000 synthetic pre-retirement customer records.

The modeling workflow removes duplicate records, excludes the customer identifier and target-derived leakage fields, engineers useful features, imputes missing values, encodes categorical variables, and prepares the model inputs.

The final modeling matrices contain:

Dataset

Rows

Features

Training

23,880

40

Test

5,970

40

The final feature set includes retained numeric predictors, engineered numeric features, and one-hot encoded categorical indicators.

Preventing target leakage
The following columns are excluded from model inputs because they are derived from the target or its relationship to the retirement goal:

Funding_Gap

Readiness_Score

RetirementReady

CustomerID is also excluded because it is an identifier, not a meaningful predictor.

The business measures are calculated after the model makes its prediction.

3. The seven notebooks
Notebook 1 — Exploratory Data Analysis
Purpose: Understand the raw dataset before building a model.

The analysis examines dataset structure, data types, missing values, duplicates, distributions, correlations, outliers, and relationships with the target. It also identifies target leakage and turns the findings into modeling decisions.

Main result: A documented understanding of the dataset and a clear plan for preprocessing and modeling.

Notebook 2 — Data Preprocessing & Feature Engineering
Purpose: Convert raw customer data into consistent model-ready features.

The notebook separates the target from the predictors, removes invalid inputs, handles missing values, encodes categorical variables, and builds engineered features. The transformations are organized into reusable pipelines so the same processing can be applied at training and prediction time.

The feature-engineering logic is maintained in src/feature_engineering.py.

Main result: Reproducible training and test data with 40 features and preprocessing components that can be reused by later notebooks.

Notebook 3 — Baseline Model Training & Evaluation
Purpose: Establish a benchmark and determine which model family works best.

The notebook compares a median baseline, Linear Regression, Ridge, Lasso, Elastic Net, Random Forest, and Gradient Boosting. It uses validation data and cross-validation for model comparison, while keeping the final test set separate for the final evaluation.

Main result: Gradient Boosting is selected as the strongest tested baseline. Its test performance was approximately:

Metric

Baseline result

R² in log space

0.896

R² in dollars

0.843

RMSE

USD 1.266M

MAE

USD 770K

Median absolute error

USD 449K

MAPE

21.3%

These results describe performance on this synthetic dataset, not on real retirement savers.

Notebook 4 — Model Improvement
Purpose: Improve the baseline and select the final model recipe.

The notebook tunes Gradient Boosting and compares alternatives, including training on the log target versus the dollar target. It also tests an additional age-based feature and applies a smearing correction when converting log predictions back to dollars.

The selected model uses Gradient Boosting with:

n_estimators = 600

learning_rate = 0.05

max_depth = 4

min_samples_leaf = 5

subsample = 0.8

Main result: The final model reduces test RMSE from approximately USD 1.266M to USD 1.231M, an improvement of about 2.75%. Aggregate prediction bias also improves after applying the smearing correction.

The final model is saved in 4/artifacts/04_final_model.joblib, with its configuration saved alongside it.

Notebook 5 — Model Evaluation & Explainability
Purpose: Understand the final model's performance, important features, and error patterns.

The notebook reloads the saved model without retraining it. It evaluates the model, examines errors across customer segments, uses permutation importance to assess feature importance, and uses SHAP to explain global patterns and individual predictions.

Important predictive features include YearsUntilRetirement, RetirementAccountBalance, and SalaryBasedContribution. The analysis also identifies segments with higher relative errors and cases where very large retirement funds are under-predicted.

Main result: A more complete view of model performance, feature importance, individual predictions, and limitations. Feature importance and SHAP explanations describe model behaviour; they do not establish causation.

Notebook 6 — Final Business Report
Purpose: Translate model predictions into business-friendly measures.

The notebook uses the final model results to create business summaries, segment summaries, customer examples, and visualizations. It calculates the funding gap and funding ratio, then applies illustrative readiness categories:

On Track: funding ratio is at least 1.00.

Needs Attention: funding ratio is at least 0.75 and below 1.00.

Significant Gap: funding ratio is below 0.75.

The 0.75 threshold is an illustrative business assumption, not a value learned by the model.

Main result: Business-facing outputs that explain how model predictions could support a retirement-planning workflow, with limitations clearly documented.

Notebook 7 — Deployment & Prediction Pipeline
Purpose: Demonstrate how the saved model can be used to predict outcomes for new customers.

The notebook generates 15 new synthetic customer records, checks the expected input schema, applies the saved preprocessing and feature-engineering pipeline, and loads the frozen final model. It does not retrain the model.

For each customer, it produces the predicted retirement fund, goal, funding gap, funding ratio, and readiness category.

Main result: A reusable prediction workflow and the files needed to build the Power BI dashboard:

7/artifacts/07_powerbi_customer_predictions.csv

7/artifacts/07_powerbi_data_dictionary.csv

The notebook includes verification checks for the prediction workflow and saved model.

4. Final model performance
The selected model is the tuned Gradient Boosting model trained on the log-transformed target, with a smearing correction applied after converting predictions back to dollars.

Its approximate test-set results are:

Metric

Final result

R² in log space

0.902

R² in dollars

0.852

RMSE

USD 1.231M

MAE

USD 753K

Median absolute error

USD 437K

MAPE

21.0%

Compared with the baseline, test RMSE improves by approximately 2.75%. The results are useful for comparing modeling approaches within this project, but they do not establish real-world predictive performance.

5. Power BI dashboard — next stage
The final stage is to turn the model outputs into an interactive dashboard. The dashboard will use the prediction CSV and its data dictionary produced by Notebook 7.

The planned pages are:

Executive Overview — customer count, average predicted fund, average retirement goal, funding ratio, projected shortfall, and readiness distribution.

Retirement Readiness — funding-ratio distribution and readiness comparisons across customer segments.

Customer View — an individual customer's profile, predicted fund, goal, funding gap, funding ratio, and readiness category.

Model & Data Insights — model performance, baseline-versus-final comparison, error patterns, and important model features.

The dashboard will focus on explaining the predictions and their business meaning. It will not retrain the model.

6. Repository structure
The repository contains the seven notebook stages, shared feature-engineering code, model artifacts, and dashboard handoff files. The key outputs include:

src/
└── feature_engineering.py

4/
└── artifacts/
    ├── 04_final_model.joblib
    └── 04_final_model_config.json

7/
└── artifacts/
    ├── 07_powerbi_customer_predictions.csv
    └── 07_powerbi_data_dictionary.csv
The repository also contains the synthetic dataset, notebook outputs, figures, and supporting artifacts generated throughout the project.

7. Limitations
Synthetic data: The dataset is generated for learning and portfolio demonstration. The model's performance should not be generalized to real retirement savers.

Not financial advice: The predictions and readiness categories are illustrative and are not a substitute for professional financial planning.

Model explanations: Feature importance and SHAP describe associations used by the model; they do not prove causal relationships.

Business thresholds: The readiness categories are business rules. In particular, the 0.75 threshold is illustrative and would require validation before use in a real service.

Prediction uncertainty: A predicted retirement fund is an estimate, not a guarantee of a future balance.

8. Project conclusion
This project demonstrates an end-to-end machine-learning workflow:

Exploratory analysis → preprocessing and feature engineering → baseline modeling → model improvement → evaluation and explainability → business reporting → prediction pipeline → Power BI dashboard.

The goal is not only to train a model, but also to understand its limitations, explain its predictions, translate its outputs into business measures, and prepare the results for an interactive dashboard.

Current status: Notebooks 1–7 are complete. The Power BI dashboard is the next stage.
