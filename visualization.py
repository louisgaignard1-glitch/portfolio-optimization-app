import plotly.graph_objects as go
import streamlit as st
import numpy as np
from scipy.optimize import minimize


def plot_efficient_frontier(result_min_var, mu, Sigma, assets):
    n = len(assets)
    target_returns = np.linspace(float(mu.min()), float(mu.max()), 40)
    frontier_vols = []
    frontier_rets = []

    for target_return in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, tr=target_return: np.dot(w, mu.values) - tr},
        )
        result = minimize(
            lambda w: np.sqrt(np.dot(w.T, np.dot(Sigma.values, w))),
            np.ones(n) / n,
            method="SLSQP",
            bounds=tuple((0, 1) for _ in assets),
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if result.success:
            vol = np.sqrt(np.dot(result.x.T, np.dot(Sigma.values, result.x)))
            frontier_vols.append(float(vol))
            frontier_rets.append(float(target_return))

    mv_weights = result_min_var.x
    mv_vol = float(np.sqrt(mv_weights @ Sigma.values @ mv_weights))
    mv_ret = float(np.dot(mv_weights, mu.values))

    fig = go.Figure()

    if frontier_vols:
        fig.add_trace(go.Scatter(
            x=frontier_vols,
            y=frontier_rets,
            mode="lines+markers",
            name="Efficient Frontier",
            line=dict(color="#00b4d8", width=2),
            marker=dict(size=5),
        ))

    fig.add_trace(go.Scatter(
        x=[mv_vol],
        y=[mv_ret],
        mode="markers",
        name="Min Variance",
        marker=dict(color="red", size=12, symbol="star"),
    ))

    fig.update_layout(
        title="Efficient Frontier (Markowitz)",
        xaxis_title="Annualised Volatility",
        yaxis_title="Annualised Return",
        xaxis=dict(tickformat=".1%"),
        yaxis=dict(tickformat=".1%"),
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", y=-0.15),
    )

    st.plotly_chart(fig, use_container_width=True, key="efficient_frontier")


def plot_sector_allocation(allocation):
    sectors = {
        "ENGI.PA": "Énergie",
        "BNP.PA": "Finance",
        "ACA.PA": "Finance",
        "GLE.PA": "Finance",
        "TTE.PA": "Énergie",
        "MC.PA": "Luxe",
        "OR.PA": "Consommation",
        "AIR.PA": "Aéronautique",
        "RNO.PA": "Automobile",
        "VK.PA": "Industrie",
        "KER.PA": "Luxe",
        "RMS.PA": "Luxe",
        "SAF.PA": "Aéronautique",
        "HO.PA": "Défense",
        "SU.PA": "Industrie",
        "CAP.PA": "Technologie",
        "STMPA.PA": "Technologie",
        "EDF.PA": "Énergie",
        "VIE.PA": "Environnement",
        "EN.PA": "Construction",
        "SAN.PA": "Santé",
        "SGO.PA": "Matériaux",
        "ORA.PA": "Télécom",
        "CA.PA": "Distribution",
        "RI.PA": "Consommation",
        "DG.PA": "Construction",
        "AI.PA": "Industrie",
    }

    sector_allocation = allocation.copy()
    sector_allocation["Secteur"] = sector_allocation.index.map(sectors).fillna("Autre")
    sector_allocation = sector_allocation.groupby("Secteur")["Poids"].sum().reset_index()

    fig = go.Figure(go.Bar(
        x=sector_allocation["Secteur"],
        y=sector_allocation["Poids"],
        marker_color="#00b4d8",
    ))

    fig.update_layout(
        title="Sector Allocation",
        xaxis_title="Sector",
        yaxis_title="Weight",
        yaxis=dict(tickformat=".0%"),
        template="plotly_dark",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True, key="sector_allocation")


def plot_correlation_matrix(returns):
    corr = returns.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation", tickformat=".1f"),
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10),
    ))

    fig.update_layout(
        title="Asset Correlation Matrix",
        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        template="plotly_dark",
        height=500,
        margin=dict(l=100, b=120),
    )

    st.plotly_chart(fig, use_container_width=True, key="correlation_matrix")
