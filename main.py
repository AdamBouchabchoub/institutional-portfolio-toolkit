"""
Institutional Portfolio Toolkit - Demo Script

Demonstrates all modules:
1. Data loading and summary statistics
2. Portfolio optimization (Equal Weight, Min Variance, Max Sharpe)
3. Performance metrics (Sharpe, Sortino, Max Drawdown, Calmar)
4. Risk analysis (VaR, CVaR, stress testing)
5. Rebalancing strategies

Author: Adam Bouchabchoub - CFA Level II
"""

import data_loader as dtld
import optimization as opt
import performance_metrics as pm
import risk_metrics as rm
import rebalancing as rb
import bond_pricing as bp
import numpy as np


def main():
    print("\n" + "="*70)
    print("INSTITUTIONAL PORTFOLIO TOOLKIT - DEMONSTRATION")
    print("="*70)
    
    # ======================== MODULE 1: DATA LOADING ========================
    print("\n" + "="*70)
    print("MODULE 1: DATA LOADING & SUMMARY STATISTICS")
    print("="*70)
    
    tickers = ['SPY', 'TLT', 'GLD', 'EEM']
    prices = dtld.fetch_prices(tickers, '2020-01-01', '2024-12-31')
    data = dtld.get_data_summary(prices)
    
    print("\nAnnualized Returns:")
    print(data['mean_returns'])
    print("\nAnnualized Volatility:")
    print(data['volatility'])
    print("\nCorrelation Matrix:")
    print(data['correlation_matrix'])
    
    # ==================== MODULE 2: PORTFOLIO OPTIMIZATION ====================
    print("\n" + "="*70)
    print("MODULE 2: PORTFOLIO OPTIMIZATION")
    print("="*70)
    
    mean_returns = data['mean_returns']
    cov_matrix = data['covariance_matrix']
    
    opt.compare_portfolios(mean_returns, cov_matrix)
    
    # Get optimized portfolios
    min_vol = opt.minimize_volatility(mean_returns, cov_matrix)
    max_sharpe = opt.maximize_sharpe(mean_returns, cov_matrix)
    
    print("\n--- Min Variance Allocation ---")
    for ticker, weight in zip(sorted(tickers), min_vol.x):
        if weight > 0.01:
            print(f"{ticker}: {weight:.1%}")
    
    print("\n--- Max Sharpe Allocation ---")
    for ticker, weight in zip(sorted(tickers), max_sharpe.x):
        if weight > 0.01:
            print(f"{ticker}: {weight:.1%}")
    
    # ==================== MODULE 3: PERFORMANCE METRICS ====================
    print("\n" + "="*70)
    print("MODULE 3: PERFORMANCE METRICS")
    print("="*70)
    
    returns = data['returns']
    
    # Equal Weight portfolio
    n = len(tickers)
    eq_weights = np.repeat(1/n, n)
    eq_returns = pm.portfolio_returns(returns, eq_weights)
    
    print("\nEqual Weight Portfolio:")
    print(f"  Sharpe Ratio:  {pm.sharpe_ratio(eq_returns):.4f}")
    print(f"  Sortino Ratio: {pm.sortino_ratio(eq_returns):.4f}")
    print(f"  Max Drawdown:  {pm.max_drawdown(eq_returns):.2%}")
    print(f"  Calmar Ratio:  {pm.calmar_ratio(eq_returns):.4f}")
    
    # ==================== MODULE 4: RISK ANALYSIS ====================
    print("\n" + "="*70)
    print("MODULE 4: RISK ANALYSIS (VaR & STRESS TESTING)")
    print("="*70)
    
    print("\nHistorical VaR (95% confidence):")
    print(rm.historical_var(returns))
    
    print("\nConditional VaR (CVaR):")
    print(rm.cvar_historic(returns))
    
    # Stress test
    weights = dict(zip(sorted(tickers), max_sharpe.x))
    covid_shock = {'SPY': -0.35, 'TLT': 0.20, 'GLD': 0.10, 'EEM': -0.40}
    
    print("\n--- Stress Test: COVID-19 Scenario ---")
    impact = rm.stress_test_scenario(weights, covid_shock)
    print(f"Max Sharpe Portfolio Impact: {impact:.2%}")
    
    # ==================== MODULE 5: REBALANCING ====================
    print("\n" + "="*70)
    print("MODULE 5: REBALANCING STRATEGIES")
    print("="*70)
    
    target_weights = {'SPY': 0.40, 'TLT': 0.30, 'GLD': 0.20, 'EEM': 0.10}
    
    print("\nComparing 3 strategies:")
    buy_hold = rb.backtest_rebalancing(returns, target_weights, threshold=1.0)
    threshold_5 = rb.backtest_rebalancing(returns, target_weights, threshold=0.05)
    
    # ==================== MODULE 6: BOND ANALYTICS ====================
    print("\n" + "="*70)
    print("MODULE 6: BOND ANALYTICS")
    print("="*70)
    
    bonds = {
        'Treasury 10Y': {'face_value': 100, 'coupon_rate': 0.04, 'ytm': 0.045, 'years_to_maturity': 10},
        'Corporate BBB': {'face_value': 100, 'coupon_rate': 0.06, 'ytm': 0.065, 'years_to_maturity': 10}
    }
    
    print("\nBond Comparison:")
    df = bp.compare_bonds(bonds)
    print(df.to_string(index=False))
    
    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nAll modules tested successfully!")
    print("\nToolkit capabilities demonstrated:")
    print("  ✓ Data loading & statistics")
    print("  ✓ Portfolio optimization (Markowitz)")
    print("  ✓ Performance metrics (Sharpe, Sortino, Drawdown)")
    print("  ✓ Risk management (VaR, CVaR, stress tests)")
    print("  ✓ Rebalancing backtests")
    print("  ✓ Bond analytics (duration, convexity)")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()