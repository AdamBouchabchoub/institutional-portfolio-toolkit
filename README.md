# Institutional Portfolio Toolkit

Professional Python toolkit for quantitative portfolio analysis and risk management.

## Features

**Chapter I: Portfolio Optimization**
- Mean-variance optimization (Markowitz)
- Minimum variance portfolio
- Maximum Sharpe ratio portfolio
- Equal-weight benchmark comparison

**Chapter III: Risk Management**
- Value at Risk (VaR): Historical, Parametric, Monte Carlo
- Conditional VaR (CVaR/Expected Shortfall)
- Stress testing with custom scenarios
- Distribution visualization

## Installation
```bash
pip install pandas numpy scipy matplotlib yfinance
```

## Quick Start
```python
import data_loader as dtld
import optimization as opt

# Fetch data
tickers = ['SPY', 'TLT', 'GLD', 'EEM']
prices = dtld.fetch_prices(tickers, '2020-01-01', '2024-12-31')
data = dtld.get_data_summary(prices)

# Optimize portfolio
opt.compare_portfolios(data['mean_returns'], data['covariance_matrix'])
```

## Project Structure
```
institutional-portfolio-toolkit/
├── data_loader.py
├── optimization.py
├── performance_metrics.py
├── risk_metrics.py
└── README.md
```

## Author

Adam Bouchabchoub - CFA Level II passed