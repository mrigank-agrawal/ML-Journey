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


## Key Concepts Covered
- Linear regression and OLS optimisation
- Feature engineering from time series data
- Train/test split without shuffling for financial data
- R² and RMSE as evaluation metrics
- Efficient Market Hypothesis — empirically validated
- Data leakage and why it matters in financial ML