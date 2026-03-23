# Stage 2 — Core ML

Building and evaluating classical machine learning models 
using real UK financial market data.

## Notebooks

### 01 - Linear Regression: HSBC Return Prediction
- Real HSBC (HSBA.L) closing price data — 1 year, 254 trading days
- Feature engineering: previous 1, 2, 3 day returns
- Train/test split respecting time order — no data leakage
- Model evaluation: R² = -0.04, RMSE = 2.25%
- **Key concepts:** OLS regression, train/test split, R², RMSE, data leakage
- **Finding:** Past returns have no predictive power — confirms the 
  Efficient Market Hypothesis. Richer features needed for real signal.
- **Next step:** Add volume, volatility and macro features to improve signal

### 02 - Feature Engineering: Beating the Basic Model
- 2 years of HSBC data with 10 engineered features
- Momentum, volatility, volume and moving average features
- **Critical lesson:** Discovered and fixed data leakage — R² dropped 
  from 0.9995 to -0.0488 after fix, proving the leakage was real
- RMSE improved from 2.25% to 1.83% despite low R²
- **Key concepts:** Feature engineering, rolling windows, data leakage, 
  model explainability, feature importance
- **Finding:** Volatility clustering confirmed — volatility_10 and 
  volatility_20 are the strongest predictors. Linear regression 
  cannot capture non-linear patterns — Random Forest next.

### 03 - Random Forest: Ensemble Learning
- 500 decision trees trained on same HSBC dataset
- Direct comparison against Linear Regression — same features, 
  different algorithm
- **Key finding:** Random Forest underperformed Linear Regression 
  on noisy financial data — bias-variance tradeoff in action
- Error distribution centred at zero — no systematic bias
- Feature importance: price vs MA dominates (mean reversion signal)
- **Key concepts:** Ensemble learning, bagging, bias-variance tradeoff,
  feature importance, n_estimators, max_depth, overfitting

  ### 04 - Classification: Predicting Stock Direction
- Binary classification — predict up or down for HSBC next day
- Two models compared: Logistic Regression vs Random Forest
- **Baseline to beat: 60.9%** — natural upward drift of markets
- Logistic Regression: 61.22% accuracy, AUC 0.5482
- Random Forest: 59.18% accuracy, AUC 0.5961
- **Key finding:** Both models biased toward up days — class 
  imbalance problem. Down day recall critically low.
- **Key concepts:** Binary classification, precision, recall, F1,
  confusion matrix, ROC-AUC, class imbalance, baseline accuracy
- **Next step:** Add macro indicators and news sentiment features,
  apply class balancing techniques

  ### 05 - XGBoost: Credit Risk Prediction
- German Credit Risk dataset — 1000 loan applicants, 20 features
- One-hot encoding of 13 categorical variables → 48 features
- XGBoost vs Random Forest — AUC 0.8056 vs 0.7924
- Early stopping — XGBoost converged in just 20 trees vs 500
- **Key finding:** Credit risk is highly predictable (AUC 0.80) 
  unlike stock prices (AUC 0.59) — measurable human behaviour 
  vs efficient markets
- Threshold tuning — moving threshold from 0.5 to 0.55 saved 
  £205,000 in bad loans — business cost translation of ML metrics
- Feature importance maps perfectly to finance intuition — 
  checking account, credit history, loan duration top predictors
- **Key concepts:** XGBoost, boosting vs bagging, early stopping,
  one-hot encoding, threshold tuning, business cost analysis,
  GDPR Right to Explanation, explainable AI

## Key Concepts Covered
- Linear regression and OLS optimisation
- Feature engineering from time series data
- Train/test split without shuffling for financial data
- R² and RMSE as evaluation metrics
- Efficient Market Hypothesis — empirically validated
- Data leakage and why it matters in financial ML