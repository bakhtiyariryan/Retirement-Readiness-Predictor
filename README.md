[README.md](https://github.com/user-attachments/files/31492236/README.md)
# Retirement Readiness Predictor

A machine-learning project that predicts a customer's **`Expected_Retirement_Fund`** from demographic, income, balance-sheet, and behavioural attributes.

The project is built as a sequence of reproducible notebooks:

- **Notebook 1 — Exploratory Data Analysis:** understand the data, identify leakage, assess data quality, and turn findings into modeling decisions.
- **Notebook 2 — Data Preprocessing & Feature Engineering:** build a leakage-safe preprocessing pipeline, create engineered features, encode categorical variables, and prepare the final modeling matrices.
- **Notebook 3 — Baseline Model Training & Evaluation:** establish a baseline, compare linear and tree-based regressors, evaluate the selected model on an untouched test set, and document the main error patterns.

The dataset is **synthetic** and is used for learning, experimentation, and portfolio demonstration. The results should not be interpreted as evidence about real retirement savers.

---

## Project status

| Stage | Status | Main result |
|---|---|---|
| **01 — EDA** | Complete | Dataset is suitable for regression after leakage removal; non-linearity and several important financial patterns were identified. |
| **02 — Preprocessing & Feature Engineering** | Complete | A reproducible pipeline produces **23,880 training rows × 40 features** and **5,970 test rows × 40 features**. |
| **03 — Baseline Model Training & Evaluation** | Complete | **Gradient Boosting** is the strongest tested baseline and is saved as a reloadable full pipeline. |
| **04 — Model Improvement** | Next | Tune the model and investigate the remaining error patterns and modeling limitations. |

---

# 1. The prediction problem

### Target

**`Expected_Retirement_Fund`**

A continuous dollar-valued estimate of the customer's projected retirement fund.

### Task

**Supervised regression**

The model receives information available about a customer and predicts the expected retirement fund.

The project evaluates predictions both in:

- **log space**, where the models are trained;
- **dollars**, where the results have direct financial meaning.

The target is strongly right-skewed, so the modeling workflow uses:

```python
y_log = np.log(y)
```

and converts predictions back to dollars with:

```python
y_pred = np.exp(y_pred_log)
```

The notebook reports both log-space and dollar-space metrics because they answer different questions.

---

# 2. Dataset

**30,000 synthetic pre-retirement customers**

The original dataset contains **32 columns**, including the target and business-derived columns identified during EDA.

The final modeling dataset is produced after:

1. removing exact duplicate records;
2. separating the target;
3. removing the identifier;
4. removing target-derived leakage columns;
5. engineering additional predictors;
6. imputing missing values;
7. encoding categorical variables;
8. scaling numeric variables where appropriate.

### Final modeling matrices

| Dataset | Rows | Features |
|---|---:|---:|
| Training | **23,880** | **40** |
| Test | **5,970** | **40** |

The final 40 features consist of:

- **17 retained numeric predictors**
- **5 engineered numeric features**
- **18 one-hot encoded categorical indicators**

The final feature count is therefore:

**17 + 5 + 18 = 40**

---

# 3. Notebook 1 — Exploratory Data Analysis

[`01_exploratory_data_analysis.ipynb`](01_exploratory_data_analysis.ipynb)

Notebook 1 answers a simple question:

> **Is this dataset ready for machine learning, and what should the model be allowed to learn from?**

The EDA covers:

| Area | Main conclusion |
|---|---|
| Data quality | Missing values are sparse; exact duplicates must be removed before splitting. |
| Target | `Expected_Retirement_Fund` is strongly right-skewed, supporting a log-target approach. |
| Leakage | `Funding_Gap`, `Readiness_Score`, and `RetirementReady` are derived from the target and must not be model inputs. |
| Identifier | `CustomerID` is removed because it is an identifier rather than a meaningful predictor. |
| Correlation | Only two feature pairs remain above \|r\| = 0.90 after the dataset revision. |
| Non-linearity | `Age` has weak Pearson correlation but a substantially stronger curved relationship with the target. |
| Missingness | The observed missingness pattern is consistent with MCAR under the notebook's permutation test. |
| Outliers | Extreme financial values are coherent and are retained rather than mechanically deleted. |
| Retirement horizon | `YearsUntilRetirement` is derived from `DesiredRetirementAge − Age`. |
| Feature engineering | Several relationships are represented more directly through engineered features. |

### Most important EDA finding: leakage

Three columns contain information derived from the target:

- `Funding_Gap`
- `Readiness_Score`
- `RetirementReady`

`Funding_Gap` is exactly:

```text
Expected_Retirement_Fund - Retirement_Fund_Goal
```

Therefore:

```text
Funding_Gap + Retirement_Fund_Goal
```

reconstructs the target exactly.

`Readiness_Score` also reconstructs the target to rounding precision, while `RetirementReady` is a direct classification of whether the funding gap is positive.

These are useful **business KPIs**, but they are not valid predictors.

The correct direction is:

```text
Customer features
       ↓
Predicted retirement fund
       ↓
Funding_Gap
Readiness_Score
RetirementReady
```

The KPIs are calculated **after** the prediction, not fed into the model.

### EDA-to-modeling decisions

The EDA led directly to the preprocessing and modeling strategy:

- remove duplicates before the split;
- remove identifier and leakage columns;
- engineer `YearsUntilRetirement`;
- handle missing values inside the pipeline;
- use one-hot encoding for categorical variables;
- scale numeric features for models that benefit from it;
- keep an unscaled pipeline for tree-based models;
- train regression models against the log-transformed target;
- test both linear and non-linear model families.

---

# 4. Notebook 2 — Data Preprocessing & Feature Engineering

[`02_data_preprocessing.ipynb`](02_data_preprocessing.ipynb)

Notebook 2 turns the EDA decisions into a reproducible preprocessing system.

The central principle is:

> **Anything learned from the data must be learned from the training data only.**

This prevents information from the test set from influencing preprocessing.

## What Notebook 2 does

### 1. Remove exact duplicates

The final modeling records are deduplicated before the train/test split.

This prevents identical records from appearing in both datasets and making the test set look artificially easy.

### 2. Remove invalid predictors

The pipeline removes:

```text
CustomerID
Funding_Gap
Readiness_Score
RetirementReady
```

The target itself is also separated from the predictors.

### 3. Create engineered features

The feature-engineering logic is stored separately in:

```text
src/feature_engineering.py
```

This keeps reusable transformation logic outside the notebook and allows the saved model pipeline to reload it later.

The pipeline creates **5 engineered features**.

### 4. Handle missing values

Missing numeric values are replaced with **training-set medians**.

The imputer is fitted on the training data and then reused unchanged on validation/test data.

### 5. Encode categorical variables

Categorical columns are converted into indicator columns.

For a categorical variable with several levels, one level is used as the reference category and the remaining levels receive their own 0/1 columns.

Unknown categories at prediction time are handled safely.

### 6. Scale numeric features

Two preprocessing variants are retained:

- **scaled** — appropriate for linear, regularized, SVM, and distance-based models;
- **unscaled** — used by tree-based models so numeric split thresholds remain in their original units.

### 7. Combine everything into a single pipeline

The preprocessing pipeline ensures that:

```text
raw data
   ↓
numeric branch ── impute → engineer → impute → scale
   ↓
categorical branch ── impute → encode
   ↓
40 final features
```

The exact same transformation sequence can therefore be applied during training and prediction.

## Final output

Notebook 2 produces:

```text
Train: 23,880 × 40
Test:   5,970 × 40
```

with:

- no missing values;
- no infinite values;
- no identifier or KPI leakage columns;
- identical train/test feature names;
- imputation statistics learned from training data only.

### Saved artifacts

Notebook 2 saves the preprocessing components required by later stages, including:

```text
artifacts/
├── preprocessor.joblib
├── preprocessor_unscaled.joblib
├── feature_names.csv
└── raw train/test splits and targets
```

---

# 5. Notebook 3 — Baseline Model Training & Evaluation

[`03_baseline_model_training.ipynb`](03_baseline_model_training.ipynb)

Notebook 3 answers:

> **How well can the first set of reasonable models predict the retirement fund, and what does their performance tell us about the next modeling step?**

The notebook deliberately starts with simple models before moving to more flexible tree ensembles.

## Evaluation strategy

The final test set remains sealed while models are compared.

The training data is split again into:

- **19,104 fitting records**
- **4,776 validation records**

The validation set is used for the initial model comparison.

Then **5-fold cross-validation** is performed on the training data.

Finally, the selected model is evaluated once on the untouched **5,970-row test set**.

The test set is therefore used only for the final performance estimate.

### Models compared

1. **Median baseline**
2. **Linear Regression**
3. **Ridge**
4. **Lasso**
5. **Elastic Net**
6. **Random Forest**
7. **Gradient Boosting**

The baseline is intentionally simple: it predicts the same median retirement fund for every customer.

A trained model should clearly outperform this constant prediction.

---

# 6. Baseline model results

## Validation performance

The validation comparison shows a clear separation between the model families.

| Model | R² | RMSE (log) | RMSE (USD) | MAE (USD) | MAPE |
|---|---:|---:|---:|---:|---:|
| **Gradient Boosting** | **0.8877** | **0.2874** | **1,245,172** | **757,712** | **21.1%** |
| Random Forest | 0.8725 | 0.3063 | 1,413,716 | 834,350 | 22.9% |
| Lasso | 0.7740 | 0.4078 | 1,808,324 | 1,109,035 | 33.9% |
| Elastic Net | 0.7749 | 0.4070 | 1,812,068 | 1,108,044 | 33.8% |
| Ridge | 0.7755 | 0.4065 | 1,820,702 | 1,108,215 | 33.8% |
| Linear Regression | 0.7755 | 0.4065 | 1,820,911 | 1,108,265 | 33.8% |
| Median baseline | −0.0175 | 0.8654 | 3,181,034 | 2,209,120 | 117.2% |

### What stands out

Gradient Boosting is the strongest model in the baseline comparison.

The notebook computes the main differences directly:

- **31.1% lower dollar RMSE than the best linear model**
- **11.9% lower dollar RMSE than Random Forest**
- **60.9% lower dollar RMSE than the median baseline**
- the four linear models are within **0.7%** of one another

This is strong evidence that the main limitation of the linear models is the **linear form itself**, rather than simply insufficient regularization.

---

# 7. Cross-validation

Five-fold cross-validation is performed using the training data only.

| Model | RMSE (log), mean ± SD | RMSE (USD), mean ± SD |
|---|---:|---:|
| **Gradient Boosting** | **0.2787 ± 0.0078** | **1,239,045 ± 48,026** |
| Random Forest | 0.2964 ± 0.0057 | 1,361,857 ± 42,200 |
| Ridge | 0.4056 ± 0.0072 | 1,809,371 ± 62,708 |

Cross-validation supports the same conclusion as the validation comparison:

**Gradient Boosting is the strongest baseline model among the tested candidates.**

The test set is not used to make this choice.

---

# 8. Final test-set result

The selected model is:

```python
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

It uses the **unscaled preprocessing pipeline**, which is appropriate for a tree-based model.

### Performance on the unseen test set

| Metric | Result |
|---|---:|
| R² — log space | **0.8962** |
| R² — dollars | **0.8429** |
| RMSE — log space | 0.2814 |
| MAE — log space | 0.2030 |
| RMSE — dollars | **USD 1,266,265** |
| MAE — dollars | **USD 769,677** |
| Median absolute error | **USD 448,701** |
| Mean absolute percentage error | **21.3%** |
| Median absolute percentage error | **15.3%** |

The model therefore explains approximately **84.3% of the variance in dollar-space test outcomes**.

The median absolute error is substantially smaller than the RMSE, which indicates that a smaller number of large errors have a meaningful effect on RMSE.

---

# 9. What the baseline model tells us

Notebook 3 produced several important findings for the next stage.

### 1. Non-linearity is predictive, not only descriptive

Notebook 1 identified a curved relationship between `Age` and the target.

Notebook 3 confirms that this matters for prediction:

**Tree ensembles reduce dollar RMSE by about 31% compared with the best linear model.**

The non-linear structure identified during EDA therefore appears in actual predictive performance.

### 2. Regularization was not the main lever

At the tested settings, Linear Regression, Ridge, Lasso, and Elastic Net perform almost identically.

All four are within **0.7%** on dollar RMSE.

This does not prove that regularization can never help. It means that, at these settings, changing the penalty is much less important than moving beyond a linear functional form.

### 3. Error is not uniform

Residual analysis shows higher error in some parts of the customer population.

Examples include:

- residual standard deviation of **0.375** in the lowest predicted quintile versus **0.246–0.268** elsewhere;
- relative error of **27.6%** in the lowest income quintile;
- relative error of **31.4%** among customers with 31+ years until retirement;
- relative error of roughly **18–22%** in several other segments.

The notebook describes these patterns without claiming a causal explanation.

### 4. Predictions are slightly low in aggregate after back-transformation

The mean predicted retirement fund is approximately **USD 3.90M**, compared with **USD 4.06M** actual.

The reason is important:

Exponentiating a log-space prediction produces a median-centred estimate under the usual log-normal interpretation of the errors. It does not recover the conditional mean.

This may be relatively unimportant for an individual prediction, but it matters if predictions are added together for portfolio-level or balance-sheet totals.

Possible future approaches include:

- a smearing correction;
- a model trained directly on the dollar target;
- a different objective.

### 5. `Retirement_Fund_Goal` has little predictive dependence in this dataset

Notebook 3 includes an ablation experiment removing `Retirement_Fund_Goal`.

The dollar RMSE changes by only about **0.5%**, while log-space R² improves slightly.

The result indicates **little predictive dependence on this variable in this synthetic dataset**.

It does not answer the separate business-process question of whether the goal would actually be known before a retirement projection is produced.

---

# 10. Error analysis

Notebook 3 does not stop at a single performance score.

It examines:

- predicted vs. actual values;
- residual distributions;
- residual behaviour across predicted-value segments;
- error by income and retirement-horizon groups;
- the effect of the `Retirement_Fund_Goal` ablation.

The analysis shows that the model works across the population, but **prediction error is not evenly distributed**.

This is important for the next stage because improving the average score alone may not improve performance for the groups where the model currently struggles most.

---

# 11. Saved baseline artifacts

Notebook 3 saves:

```text
artifacts/
├── 03_baseline_model.joblib
└── 03_baseline_model_results.csv
```

The model artifact contains the full prediction pipeline.

Because the pipeline contains the custom feature-engineering function, `src/feature_engineering.py` must remain importable when the saved model is reloaded.

The reload was verified in a fresh Python process, including an element-wise prediction comparison with a maximum difference of:

```text
0.0
```

This confirms that the saved artifact reproduces the notebook's predictions.

---

# 12. Repository structure

The project now follows this structure:

```text
.
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_baseline_model_training.ipynb
│
├── src/
│   └── feature_engineering.py
│
├── artifacts/
│   ├── preprocessor.joblib
│   ├── preprocessor_unscaled.joblib
│   ├── feature_names.csv
│   ├── 03_baseline_model.joblib
│   └── 03_baseline_model_results.csv
│
├── figures/
│   ├── EDA figures
│   └── Notebook 3 model/evaluation figures
│
├── retirement_dataset_v2.csv
└── README.md
```

---

# 13. Key figures

Notebook 1 contains the main EDA figures, including:

- target distribution;
- missingness;
- correlation structure;
- age non-linearity;
- outlier analysis;
- retirement horizon;
- leakage assessment;
- ML readiness;
- business insights.

Notebook 3 adds four model-evaluation figures:

```text
03_model_comparison.png
03_predicted_vs_actual.png
03_residuals.png
03_error_by_segment.png
```

Together, the figures show the progression from:

**understanding the data → preparing the data → evaluating predictive performance.**

---

# 14. Important limitations

### Synthetic data

The entire project uses synthetic data.

The numerical results describe this generated dataset. They are not claims about real retirement savers.

### Validation preprocessing

Sections 5–11 of Notebook 3 use a validation split carved from the training data. The preprocessing statistics for that comparison were learned from all 23,880 training rows, so those validation scores are mildly optimistic.

The five-fold cross-validation in Section 12 is stricter because the complete preprocessing pipeline is refitted inside every fold.

### Log-target back-transformation

The model is trained in log space and predictions are exponentiated back to dollars.

This is useful for relative-error behaviour, but it can underestimate aggregate totals because the exponentiated prediction is median-centred rather than mean-centred.

### Model tuning

Notebook 3 establishes a **baseline**. The Gradient Boosting hyperparameters have not been exhaustively tuned.

Therefore, the result should be interpreted as:

> **the performance of the selected baseline configuration**, not the theoretical maximum performance of Gradient Boosting on this dataset.

---

# 15. Current project conclusion

The project has now moved beyond exploratory analysis.

The EDA established **what the data contains and what the model should be allowed to see**.

The preprocessing notebook turned those decisions into a **reproducible 40-feature pipeline**.

The baseline modeling notebook showed that:

> **Gradient Boosting is substantially stronger than the tested linear models on this dataset, confirming that the non-linear structure identified during EDA has real predictive value.**

The current test performance is:

**R² = 0.843 in dollars**  
**RMSE ≈ USD 1.27M**  
**MAE ≈ USD 770K**  
**Median absolute error ≈ USD 449K**  
**MAPE = 21.3%**

The next stage should focus on **improving and stress-testing the baseline**, rather than simply adding more models without understanding the remaining errors.

---

## Next step

### Notebook 4 — Model Improvement

The baseline has identified the main areas worth investigating next:

1. **Hyperparameter tuning** for Gradient Boosting and other strong candidates.
2. Investigate whether tuning regularization or model complexity changes the current ranking.
3. Address the **low-tail and segment-specific error patterns** identified in Notebook 3.
4. Investigate the approximately **4% aggregate shortfall** caused by the log-space back-transformation.
5. Compare appropriate alternatives if portfolio-level totals are an important use case.
6. Keep the test set protected until the final comparison.

The goal is not simply to maximize a single score. The next notebook should determine whether the baseline can be improved **without sacrificing methodological discipline or interpretability**.
