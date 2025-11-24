import numpy as np
from scipy.optimize import minimize

def portfolio_return(mean_returns, weights):
    """
    Calculate portfolio return.
    """
    portfolio_return = mean_returns @ weights
    return portfolio_return


def equal_weight_portfolio(mean_returns):
    """
    Calculate equal weight portfolio return.
    """
    n = len(mean_returns)
    weights = np.repeat(1/n, n)
    equal_w_portfolio_return = portfolio_return(mean_returns, weights)
    return equal_w_portfolio_return

def portfolio_volatility(weights, cov_matrix):
    """
    Calculate portfolio volatility.
    """
    portfolio_vola = np.sqrt(weights @ cov_matrix @ weights)
    return portfolio_vola

def portfolio_sharpe(mean_returns, weights, cov_matrix, risk_free=0.025):
    """
    Calculate Sharpe Ratio.
    """
    ret = portfolio_return(mean_returns, weights)
    vol = portfolio_volatility(weights, cov_matrix)
    return (ret - risk_free) / vol

from scipy.optimize import minimize

def minimize_volatility(mean_returns, cov_matrix):
    """
    Find minimum Volatility portfolio.
    """
    n = len(mean_returns)
    
    def objective(weights):
        return portfolio_volatility(weights, cov_matrix)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0,1) for _ in range(n))
    x0 = np.repeat(1/n, n)

    result = minimize(objective, x0, method='SLSQP', constraints=constraints, bounds=bounds)
    return result 

def maximize_sharpe(mean_returns, cov_matrix, risk_free=0.025):
    """
    Find maximum Sharpe ratio portfolio.
    """
    n = len(mean_returns)
    
    def objective(weights):
        sharpe = portfolio_sharpe(mean_returns, weights, cov_matrix, risk_free)
        return -sharpe
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1} 
    bounds = tuple((0,1) for _ in range(n))
    x0 = np.repeat(1/n, n)
    
    result = minimize(objective, x0, method='SLSQP', constraints=constraints, bounds=bounds)
    return result


def compare_portfolios(mean_returns, cov_matrix, risk_free=0.025):
    """
    Compare 3 strategies: Equal Weight, Min Variance, Max Sharpe.
    """
    n= len(mean_returns)
    eq_weights = np.repeat(1/n, n)
    eq_return = portfolio_return(mean_returns, eq_weights)
    eq_vol = portfolio_volatility(eq_weights, cov_matrix)
    eq_sharpe = portfolio_sharpe(mean_returns, eq_weights, cov_matrix, risk_free)

    min_vol = minimize_volatility(mean_returns, cov_matrix)
    mv_return = portfolio_return(mean_returns, min_vol.x)
    mv_vol = min_vol.fun
    mv_sharpe = portfolio_sharpe(mean_returns, min_vol.x, cov_matrix, risk_free)

    max_sharpe = maximize_sharpe(mean_returns, cov_matrix, risk_free)
    ms_return = portfolio_return(mean_returns, max_sharpe.x)
    ms_vol = portfolio_volatility(max_sharpe.x, cov_matrix)
    ms_sharpe = -max_sharpe.fun

    print("EQUAL WEIGHT:")
    print(f"  Return: {eq_return:.2%}, Vol: {eq_vol:.2%}, Sharpe: {eq_sharpe:.4f}")
    print("\nMIN VARIANCE:")
    print(f"  Return: {mv_return:.2%}, Vol: {mv_vol:.2%}, Sharpe: {mv_sharpe:.4f}")
    print("\nMAX SHARPE:")
    print(f"  Return: {ms_return:.2%}, Vol: {ms_vol:.2%}, Sharpe: {ms_sharpe:.4f}")


if __name__ == "__main__":
    # Test code
    import data_loader as dtld
    
    tickers = ['SPY', 'TLT', 'GLD', 'EEM']
    prices = dtld.fetch_prices(tickers, '2020-01-01', '2024-12-31')
    data = dtld.get_data_summary(prices)
    
    mean_returns = data['mean_returns']
    cov_matrix = data['covariance_matrix']
    
    print("\n=== PORTFOLIO COMPARISON ===")
    compare_portfolios(mean_returns, cov_matrix)
