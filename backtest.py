import pandas as pd

def backtest_portfolio(data, allocation):
    returns = data.pct_change().dropna()
    weights = allocation['Poids'].reindex(returns.columns).fillna(0)
    portfolio_returns = returns.dot(weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    return pd.DataFrame(cumulative_returns, columns=['Cumulative Returns'])
