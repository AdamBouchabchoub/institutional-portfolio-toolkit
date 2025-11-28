# Institutional Portfolio Toolkit

Professional Python toolkit for quantitative portfolio analysis and risk management.

## Features

**Portfolio Optimization**
- Mean-variance optimization (Markowitz)
- Minimum variance portfolio
- Maximum Sharpe ratio portfolio
- Equal-weight benchmark comparison

**Performance Metrics**
- Sharpe, Sortino, Calmar Ratios
- Maximum Drawdown analysis
- Annualized returns & volatility

**Risk Management**
- Value at Risk (VaR): Historical, Parametric, Monte Carlo
- Conditional VaR (CVaR/Expected Shortfall)
- Stress testing with custom scenarios
- Distribution visualization

**Portfolio Rebalancing**
- Drift detection & threshold-based rebalancing
- Transaction cost modeling
- Backtesting framework

**Bond Analytics**
- Bond pricing (clean, dirty, accrued interest)
- Duration : Macaulay & Modified
- Convexity measurement
- Multi-bond comparison

## Installation
```bash
pip install pandas numpy scipy matplotlib yfinance
```

## Quick Start
```python
import data_loader as dtld
import optimization as opt


tickers = ['SPY', 'TLT', 'GLD', 'EEM']
prices = dtld.fetch_prices(tickers, '2020-01-01', '2024-12-31')
data = dtld.get_data_summary(prices)


opt.compare_portfolios(data['mean_returns'], data['covariance_matrix'])
```
## Run Demo
```bash
python main.py
```

## Project Structure
```
institutional-portfolio-toolkit/
├── data_loader.py
├── optimization.py
├── performance_metrics.py
├── risk_metrics.py
├── rebalancing.py
├── bond_pricing.py
├── main.py
├── requirements.txt
└── README.md
```

## Author

Adam Bouchabchoub