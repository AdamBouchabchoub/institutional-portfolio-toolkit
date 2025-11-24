import pandas as pd
import numpy as np

def historical_var(returns, confidence=0.95):
    """
    Calculate Historic VaR from actual data (no assumptions).
      - Single Asset/Portfolio return (Series) : return single VaR.
      - Multiple assets (DataFrame) : return VaR for each.
    """
    alpha = (1-confidence) * 100
    
    if isinstance(returns, pd.Series):
        var = -np.percentile(returns, alpha)
    else:
        var = -np.percentile(returns, alpha, axis=0)
        var = pd.Series(var, index=returns.columns)
        
    return var
    
def parametric_var(returns, confidence=0.95):
    """
    Calculate parametric VaR (Gaussian) assuming normal distrib.
      - Single Asset/Portfolio return (Series) : return single VaR.
      - Multiple assets (DataFrame) : return VaR for each.
    """
    from scipy import stats
    z_score = stats.norm.ppf(1 - confidence)
    
    if isinstance(returns, pd.Series):
        mean = returns.mean()
        std = returns.std()
        var = -(mean +  z_score*std)
    else:
        mean = returns.mean()
        std = returns.std()
        var = -(mean + z_score*std)
        var = pd.Series(var, index=returns.columns)

    return var

    
def monte_carlo_var(returns, confidence=0.95, n_scenarios=10000):
    """
    Calculate Monte Carlo VaR by simulating future returns.
      - Single Asset/Portfolio return (Series) : return single VaR.
      - Multiple assets (DataFrame) : return VaR for each.
    """
    alpha = (1 - confidence) * 100
    
    if isinstance(returns, pd.Series):
        mean = returns.mean()
        std = returns.std()
        simulated_returns = np.random.normal(mean, std, n_scenarios)
        var = -np.percentile(simulated_returns, alpha)

    else:
        mean = returns.mean()
        std = returns.std()
        var_list = []
        for col in returns.columns:
            simulated_returns = np.random.normal(mean[col], std[col], n_scenarios)
            var = -np.percentile(simulated_returns, alpha)
            var_list.append(var)
            
        var = pd.Series(var_list, index=returns.columns)

    return var

def cvar_historic(returns, confidence=0.95):
    """
    Calculate Conditional VaR for Series and DataFrame.
    """
    var = historical_var(returns, confidence=confidence)
    left_tail = returns[returns <= -var]
    c_var = -left_tail.mean()
    return c_var

def plot_var_distribution(returns, confidence=0.95):
    """
    Plot return distribution with VaR and CVaR lines.
    
    - Series: Single plot
    - DataFrame: Subplots for each asset
    """
    import matplotlib.pyplot as plt
    
    if isinstance(returns, pd.Series):
        var = historical_var(returns, confidence)
        cv = cvar_historic(returns, confidence)
        
        plt.figure(figsize=(10, 6))
        plt.hist(returns, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        plt.axvline(-var, color='red', linewidth=2, linestyle='--', label=f'VaR: {var:.2%}')
        plt.axvline(-cv, color='darkred', linewidth=3, label=f'CVaR: {cv:.2%}')
        plt.axvspan(returns.min(), -var, alpha=0.3, color='red', label='Tail Risk')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.title(returns.name if returns.name else 'Portfolio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    else:
        n = len(returns.columns)
        rows = (n + 1) // 2
        
        fig, axes = plt.subplots(rows, 2, figsize=(14, 5*rows))
        axes = axes.flatten() if n > 1 else [axes]
        
        for i, col in enumerate(returns.columns):
            var = historical_var(returns[col], confidence)
            cv = cvar_historic(returns[col], confidence)
            
            axes[i].hist(returns[col], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
            axes[i].axvline(-var, color='red', linewidth=2, linestyle='--', label=f'VaR: {var:.2%}')
            axes[i].axvline(-cv, color='darkred', linewidth=3, label=f'CVaR: {cv:.2%}')
            axes[i].axvspan(returns[col].min(), -var, alpha=0.3, color='red')
            axes[i].set_xlabel('Returns')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(col)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        for i in range(n, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

def stress_test_scenario(portfolio_weights, shock_dict):
    """
    Apply stress shocks to portfolio and calculate impact.
    Output: portfolio return under stress test scenario.
    """
    if isinstance(portfolio_weights, dict):
        weights = pd.Series(portfolio_weights)
    else:
        weights = portfolio_weights

    if isinstance(shock_dict, dict):
        shocks = pd.Series(shock_dict)
    else:
        shocks = shock_dict

    weights = weights[shocks.index]
    portfolio_impact = (weights*shocks).sum()
    
    return portfolio_impact     

if __name__ == "__main__":
    import data_loader as dtld
    
    # Fetch data
    tickers = ['SPY', 'TLT', 'GLD', 'EEM']
    prices = dtld.fetch_prices(tickers, '2015-01-01', '2024-12-31')
    returns = dtld.calculate_returns(prices)
    
    print("\n" + "="*60)
    print("RISK METRICS ANALYSIS")
    print("="*60)
    
    # VaR All Assets
    print("\nHistorical VaR - All Assets:")
    print(historical_var(returns))
    
    print("\nParametric VaR - All Assets:")
    print(parametric_var(returns))
    
    print("\nMonte Carlo VaR - All Assets:")
    print(monte_carlo_var(returns))
    
    # CVaR
    print("\nCVaR - All Assets:")
    print(cvar_historic(returns))
    
    # Stress Testing
    print("\n" + "-"*60)
    print("STRESS TEST SCENARIOS")
    
    weights = {'SPY': 0.25, 'TLT': 0.25, 'GLD': 0.25, 'EEM': 0.25}
    
    crisis_2008 = {'SPY': -0.40, 'TLT': 0.15, 'GLD': 0.05, 'EEM': -0.50}
    covid = {'SPY': -0.35, 'TLT': 0.20, 'GLD': 0.10, 'EEM': -0.40}
    inflation = {'SPY': -0.20, 'TLT': -0.15, 'GLD': 0.05, 'EEM': -0.25}
    
    print(f"\n2008 Crisis Impact: {stress_test_scenario(weights, crisis_2008):.2%}")
    print(f"COVID Crash Impact: {stress_test_scenario(weights, covid):.2%}")
    print(f"Inflation Impact:   {stress_test_scenario(weights, inflation):.2%}")
    
    # Plots
    print("\nGenerating distribution plots...")
    plot_var_distribution(returns)