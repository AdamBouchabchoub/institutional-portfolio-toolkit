import pandas as pd
import numpy as np
import performance_metrics as pm

def calculate_drift(current_weights, target_weights):
    """
    Calculate portfolio drift from target allocation.
    
    Parameters:
    -----------
    current_weights : dict or pd.Series
        Current portfolio weights
    target_weights : dict or pd.Series
        Target portfolio weights
        
    Returns:
    --------
    dict : Contains 'drift' (absolute deviation per asset) and 'total_drift' (sum)
    """
    current = pd.Series(current_weights)
    target = pd.Series(target_weights)
    drift = np.abs(current - target)
    total_drift = drift.sum()

    return {'drift': drift, 'total_drift': total_drift}

def calculate_portfolio_value(returns, weights, initial_value=100000):
    """
     Calculate portfolio value over time given returns and fixed weights.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns for each asset
    weights : dict or pd.Series
        Portfolio weights
    initial_value : float, optional
        Starting portfolio value (default: 100000)
        
    Returns:
    --------
    Portfolio value over time
    """
    weights = pd.Series(weights)
    weights = weights[returns.columns]
    
    portfolio_returns = returns @ weights
    cumulative_growth = (1 + portfolio_returns).cumprod()
    portfolio_value = initial_value * cumulative_growth

    return portfolio_value

def calculate_actual_weights(returns, initial_weights):
    """
    Calculate how portfolio weights drift over time without rebalancing.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns for each asset
    initial_weights : dict or pd.Series
        Starting portfolio weights
        
    Returns:
    --------
    Actual weights over time 
    """
    weights = pd.Series(initial_weights)
    weights = weights[returns.columns]

    cumulative_growth = (1 + returns).cumprod()
    asset_values = cumulative_growth * weights
    total_value = asset_values.sum(axis=1)
    actual_weights = asset_values.div(total_value, axis=0)

    return actual_weights

def rebalance_portfolio(current_weights, target_weights, portfolio_value, trading_cost=0.001):
    """
    Execute rebalancing trade and calculate transaction costs.
    
    Parameters:
    -----------
    current_weights : pd.Series or dict
        Current portfolio allocation
    target_weights : pd.Series or dict
        Target portfolio allocation
    portfolio_value : float
        Current total portfolio value
    trading_cost : float, optional
        Transaction cost as % of traded amount (default: 0.1%)
        
    Returns:
    --------
    Contains trades (% and $), total cost, new weights
    """
    current_weights = pd.Series(current_weights)
    target_weights = pd.Series(target_weights)

    trades_pct = target_weights - current_weights
    trades_dollar = trades_pct * portfolio_value

    total_trading_cost = trades_dollar.abs().sum() * trading_cost

    return {'current_weights': current_weights,
            'trades_pct': trades_pct,
            'trades_dollar': trades_dollar,
            'total_trading_cost': total_trading_cost,
            'new_weights': target_weights} 

def backtest_rebalancing(returns, target_weights, threshold=0.05, trading_cost=0.001, initial_value=100000):
    """
     Backtest portfolio rebalancing strategy with drift threshold.
    
    Rebalances when total drift exceeds threshold (e.g., 5%).
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns for each asset
    target_weights : dict or pd.Series
        Target portfolio allocation
    threshold : float, optional
        Drift threshold to trigger rebalancing (default: 5%)
    trading_cost : float, optional
        Transaction cost as % of trade (default: 0.1%)
    initial_value : float, optional
        Starting portfolio value (default: 100000)

    Returns:
    --------
    Backtest results including portfolio value, rebalance dates, 
    costs, performance metrics (return, Sharpe, max drawdown)
    """
    target_weights = pd.Series(target_weights)
    target_weights = target_weights[returns.columns]

    portfolio_value = pd.Series(index=returns.index, dtype=float)
    portfolio_value.iloc[0] = initial_value

    current_weights = target_weights.copy()
    rebalance_dates = []
    total_cost = 0

    for i in range(1, len(returns)):
        daily_return = (returns.iloc[i] * current_weights).sum()
        portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1+daily_return)
        asset_growth = 1 + returns.iloc[i]
        current_weights = (current_weights * asset_growth) / (current_weights * asset_growth).sum()

        drift = calculate_drift(current_weights, target_weights)
        if drift['total_drift'] > threshold:
            result = rebalance_portfolio(current_weights, target_weights, portfolio_value.iloc[i], trading_cost)
            portfolio_value.iloc[i] -= result['total_trading_cost']
            total_cost += result['total_trading_cost']
            current_weights = target_weights.copy()
            rebalance_dates.append(returns.index[i])

    final_return = (portfolio_value.iloc[-1]/initial_value) - 1

    portfolio_returns = portfolio_value.pct_change().dropna()
    volatility = pm.annualized_volatility(portfolio_returns)
    sharpe = pm.sharpe_ratio(portfolio_returns)
    max_dd = pm.max_drawdown(portfolio_returns)

    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS - Threshold: {threshold:.1%}")
    print(f"{'='*60}")
    print(f"Initial value:        ${initial_value:,.0f}")
    print(f"Final value:          ${portfolio_value.iloc[-1]:,.0f}")
    print(f"Total return:         {final_return:.2%}")
    print(f"Number of rebalances: {len(rebalance_dates)}")
    print(f"Total cost:          ${total_cost:,.2f}")
    print(f"Volatility:             {volatility: .2%}")
    print(f"Sharpe Ratio:           {sharpe: .3f}")
    print(f"Max drawdown:            {max_dd: .2%}")

    return {
        'portfolio_value': portfolio_value,
        'rebalance_dates': rebalance_dates,
        'total_cost': total_cost,
        'n_rebalances': len(rebalance_dates),
        'final_return': final_return,
        'volatility':   volatility,
        'sharpe_ratio':  sharpe,
        'max_drawdown': max_dd
    }

def plot_rebalancing_comparison(buy_hold, threshold_1, threshold_2):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(18,6))
    plt.plot(buy_hold['portfolio_value'], label='Buy & Hold', linewidth=2)
    plt.plot(threshold_1['portfolio_value'], label='5% Threshold', linewidth=1)
    plt.plot(threshold_2['portfolio_value'], label='10% Threshold', linewidth=1)   
    plt.legend()
    plt.title('Portfolio Value Comparison')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    import data_loader as dtld
    
    # Fetch data
    tickers = ['SPY', 'TLT', 'GLD', 'EEM']
    prices = dtld.fetch_prices(tickers, '2015-01-01', '2024-12-31')
    returns = dtld.calculate_returns(prices)
    
    # Target weights
    target = {'SPY': 0.40, 'TLT': 0.30, 'GLD': 0.20, 'EEM': 0.10}
    
    print("\n" + "="*70)
    print("REBALANCING STRATEGY COMPARISON")
    print("="*70)
    
    # Test 3 strategies
    buy_hold = backtest_rebalancing(returns, target, threshold=1.0)
    threshold_5 = backtest_rebalancing(returns, target, threshold=0.05)
    threshold_10 = backtest_rebalancing(returns, target, threshold=0.10)
    plot_rebalancing_comparison(buy_hold, threshold_5, threshold_10)

