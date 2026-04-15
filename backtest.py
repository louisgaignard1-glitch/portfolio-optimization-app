import plotly.graph_objects as go
import streamlit as st


def backtest_portfolio(data, allocation):
    returns = data.pct_change().dropna()
    weights = allocation["Poids"].reindex(returns.columns).fillna(0)
    portfolio_returns = returns.dot(weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            mode="lines",
            name="Portfolio",
            line=dict(color="#00b4d8", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,180,216,0.10)",
        )
    )

    # Reference line at 1 (initial investment)
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial value",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title="Historical Cumulative Performance",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (base 1)",
        yaxis=dict(tickformat=".2f"),
        template="plotly_dark",
        height=450,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Return the series in case the caller still needs it
    return cumulative_returns
