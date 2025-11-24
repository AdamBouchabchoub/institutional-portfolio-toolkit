import pandas as pd
import numpy as np

def calculate_returns(prices):
    """
    Calculate daily returns from prices.
    """
    returns = prices.pct_change().dropna()
    return returns

def annualized_returns(returns, period_per_year=252):
    """
    Calculate annualized return (CAGR).
    """
    n_periods = len(returns)
    annualized_returns = (1+returns).prod()**(period_per_year/n_periods) - 1
    return annualized_returns

def annualized_volatility(returns, period_per_year=252):
    """
    Calculate annualized volatility from daily returns.
    """
    annualized_volatility = returns.std() * np.sqrt(period_per_year)
    return annualized_volatility

def semi_deviation(returns, period_per_year=252):
    """
    Calculate Semi-deviation (Vol of negatif returns). 
    """
    neg_returns = returns[returns<0]
    semi_dev = neg_returns.std() * np.sqrt(period_per_year)
    return semi_dev
    

def sharpe_ratio(returns, risk_free=0.025, period_per_year=252):
    """
    Calculate Sharpe ratio for each asset (Excess return / risk).
    """
    excess_return = (annualized_returns(returns, period_per_year) - risk_free) 
    vol = annualized_volatility(returns, period_per_year)
    sharpe_ratio = excess_return / vol
    return sharpe_ratio

def sortino_ratio(returns, risk_free=0.025, period_per_year=252):
    """
    Calculate Sortino ratio for each asset (Excess return / downside risk).
    """
    excess_return = annualized_returns(returns, period_per_year) - risk_free
    downside_risk = semi_deviation(returns, period_per_year)
    sortino_ratio = excess_return / downside_risk
    return sortino_ratio

def max_drawdown(returns):
    """
    Calculate maximum drawdown (worst peak-to-trough decline).
    """
    cum_wealth = (1+returns).cumprod()
    peaks = cum_wealth.cummax()
    dd = (cum_wealth - peaks) / peaks
    max_dd = dd.min()
    return max_dd

def calmar_ratio(returns, period_per_year=252):
    """
    Calculate Calmar ratio (return / |max drawdown|).
    """
    ann_return = annualized_returns(returns, period_per_year)
    max_dd = max_drawdown(returns)
    calmar_ratio = ann_return / abs(max_dd)
    return calmar_ratio

def portfolio_returns(returns, weights):
    """
    Calculate daily portfolio returns from asset returns and weights.
    """
    return returns @ weights



if __name__ == "__main__":
    import data_loader as dtld
    import optimization as opt
    
    tickers = ['SPY', 'TLT', 'GLD', 'EEM']
    prices = dtld.fetch_prices(tickers, '2020-01-01', '2024-12-31')
    returns = calculate_returns(prices)
    data = dtld.get_data_summary(prices)
    
    mean_returns = data['mean_returns']
    cov_matrix = data['covariance_matrix']
    
    # Equal Weight
    n = len(mean_returns)
    eq_weights = np.repeat(1/n, n)
    eq_pf_returns = portfolio_returns(returns, eq_weights)
    
    # Min Variance
    min_vol = opt.minimize_volatility(mean_returns, cov_matrix)
    mv_pf_returns = portfolio_returns(returns, min_vol.x)
    
    # Max Sharpe
    max_sh = opt.maximize_sharpe(mean_returns, cov_matrix)
    ms_pf_returns = portfolio_returns(returns, max_sh.x)
    
    print("\n=== PORTFOLIO METRICS COMPARISON ===")
    
    print("\nEQUAL WEIGHT:")
    print(f"  Sharpe: {sharpe_ratio(eq_pf_returns):.4f}")
    print(f"  Sortino: {sortino_ratio(eq_pf_returns):.4f}")
    print(f"  Max DD: {max_drawdown(eq_pf_returns):.2%}")
    
    print("\nMIN VARIANCE:")
    print(f"  Sharpe: {sharpe_ratio(mv_pf_returns):.4f}")
    print(f"  Sortino: {sortino_ratio(mv_pf_returns):.4f}")
    print(f"  Max DD: {max_drawdown(mv_pf_returns):.2%}")
    
    print("\nMAX SHARPE:")
    print(f"  Sharpe: {sharpe_ratio(ms_pf_returns):.4f}")
    print(f"  Sortino: {sortino_ratio(ms_pf_returns):.4f}")
    print(f"  Max DD: {max_drawdown(ms_pf_returns):.2%}")