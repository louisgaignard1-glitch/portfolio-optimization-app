import plotly.graph_objects as go
import streamlit as st
import numpy as np
from scipy.optimize import minimize


def plot_efficient_frontier(result_min_var, mu, Sigma, assets):
    target_returns = np.linspace(mu.min(), mu.max(), 20)
    frontier_vols = []
    for target_return in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, tr=target_return: np.dot(w, mu) - tr}
        )
        result = minimize(
            lambda w: np.sqrt(np.dot(w.T, np.dot(Sigma, w))),
            np.ones(len(assets)) / len(assets),
            method='SLSQP',
            bounds=tuple((0, 1) for _ in assets),
            constraints=constraints
        )
        if result.success:
            frontier_vols.append(np.sqrt(np.dot(result.x.T, np.dot(Sigma, result.x))))
        else:
            print(f"Optimisation échouée pour le rendement cible {target_return}")
            frontier_vols.append(np.nan)  # ou une valeur par défaut

    # Supprimez les valeurs NaN si nécessaire
    frontier_vols = [vol for vol in frontier_vols if not np.isnan(vol)]
    target_returns = target_returns[:len(frontier_vols)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier_vols,
        y=target_returns,
        mode='lines+markers',
        name='Frontière efficace'
    ))
    fig.update_layout(
        title="Frontière efficace du portefeuille",
        xaxis_title="Volatilité annualisée (écart-type)",
        yaxis_title="Rendement annualisé",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)


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
