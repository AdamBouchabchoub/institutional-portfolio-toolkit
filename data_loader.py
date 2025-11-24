import pandas as pd
import numpy as np
import yfinance as yf

def fetch_prices(tickers, start_date, end_date):
    """
    Download closing prices from Yahoo Finance.
    """
    prices = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()
    return prices

def calculate_returns(prices):
    """
    Calculate daily returns from prices.
    """
    returns = prices.pct_change().dropna()
    return returns

def calculate_mean_returns(returns, annualize=True):
    """
    Calculate average return for each asset.
    On a daily or annualized basis.
    """
    mean_returns = returns.mean()
    if annualize:
        mean_returns = mean_returns * 252
    return mean_returns

def calculate_covariance_matrix(returns, annualize=True):
    """
    Calculate covariance matrix.
    On a daily or annualized basis.
    """
    cov_matrix = returns.cov()
    if annualize:
        cov_matrix = cov_matrix * 252
    return cov_matrix

def calculate_correlation_matrix(returns):
    """
    Calculate correlation matrix.
    """
    correlation = returns.corr()
    return correlation

def calculate_volatility(returns, annualize=True):
    """
    Calculate volatility (standard deviation).
    On a daily or annualized basis.
    """
    volatility = returns.std()
    if annualize:
        volatility = volatility* np.sqrt(252)
    return volatility

def get_data_summary(prices):
    """
    Calculate all statistics for the given price data.
    
    Returns dictionary with: prices, returns, mean_returns, 
    covariance_matrix, correlation_matrix, volatility
    """
    returns = calculate_returns(prices)
    mean_returns = calculate_mean_returns(returns, annualize=True)
    cov_matrix = calculate_covariance_matrix(returns, annualize=True)
    corr_matrix = calculate_correlation_matrix(returns)
    volatility = calculate_volatility(returns, annualize=True)
    
    summary = {'prices': prices,
               'returns': returns,
               'mean_returns': mean_returns,
               'covariance_matrix': cov_matrix,
               'correlation_matrix': corr_matrix,
               'volatility': volatility}
    return summary


if __name__ == "__main__":
    tickers = ['SPY', 'GLD', 'TLT', 'EEM']
    prices = fetch_prices(tickers, '2020-01-01', '2024-12-31')
    data = get_data_summary(prices)
    
    print("Mean Returns:")
    print(data['mean_returns'])
