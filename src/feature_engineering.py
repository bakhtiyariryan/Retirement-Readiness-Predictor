"""Feature engineering for the Retirement Readiness project.

These functions live in a module rather than inside a notebook so that a
preprocessing pipeline saved with joblib can be reloaded in any Python session,
including the one that serves predictions.
"""

import numpy as np

# Features created by add_engineered_features(), in the order they are added.
ENGINEERED_FEATURES = [
    "YearsUntilRetirement",
    "CareerStartAge",
    "SalaryBasedContribution",
    "RealExpectedReturn",
    "DebtToIncomeRatio",
]

# Each of these is an exact linear component of an engineered feature above.
# Keeping both would put an exact linear dependency into the feature matrix.
SOURCE_COLUMNS_TO_DROP = [
    "DesiredRetirementAge",   # YearsUntilRetirement = DesiredRetirementAge - Age
    "YearsExperience",        # CareerStartAge       = Age - YearsExperience
    "ExpectedInflation",      # RealExpectedReturn   = ExpectedAnnualReturn - ExpectedInflation
]


def add_engineered_features(numeric_data):
    """Add five engineered features and drop their exact linear sources.

    Expects a DataFrame of imputed numeric columns. Every calculation is
    row-wise, so the function holds no fitted state and cannot move
    information between the training and test sets.
    """
    data = numeric_data.copy()

    # Years of compounding left before the customer's target retirement date.
    data["YearsUntilRetirement"] = data["DesiredRetirementAge"] - data["Age"]

    # Age at which the customer entered the workforce.
    data["CareerStartAge"] = data["Age"] - data["YearsExperience"]

    # Salary-based savings proxy: the share of salary directed to savings.
    data["SalaryBasedContribution"] = data["AnnualSalary"] * data["SavingsRate"]

    # Expected growth net of expected inflation.
    data["RealExpectedReturn"] = (
        data["ExpectedAnnualReturn"] - data["ExpectedInflation"]
    )

    # Mortgage balance relative to salary. Customers with no salary have no
    # defined ratio, so the denominator is marked missing instead of dividing
    # by zero. The imputer that follows in the pipeline fills the result.
    salary = data["AnnualSalary"].replace(0, np.nan)
    data["DebtToIncomeRatio"] = data["MortgageBalance"] / salary

    return data.drop(columns=SOURCE_COLUMNS_TO_DROP)


def engineered_feature_names(transformer, input_features):
    """Report the column names add_engineered_features() produces.

    scikit-learn cannot inspect a plain function, so FunctionTransformer is
    given this callback to keep get_feature_names_out() working.
    """
    kept_columns = []
    for name in input_features:
        if name not in SOURCE_COLUMNS_TO_DROP:
            kept_columns.append(str(name))

    return kept_columns + ENGINEERED_FEATURES
