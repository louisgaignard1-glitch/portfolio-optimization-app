import plotly.graph_objects as go
import streamlit as st
import numpy as np
from scipy.optimize import minimize

def plot_efficient_frontier(result_min_var, mu, Sigma, assets):
    # Vérifiez que mu et Sigma sont valides
    if mu.isnull().any() or Sigma.isnull().any().any():
        st.error("Les données d'entrée ne sont pas valides.")
        return

    # Définissez une plage de rendements cibles réalistes
    min_return = min(mu.min(), 0)
    max_return = max(mu.max(), 0.5)
    target_returns = np.linspace(min_return, max_return, 20)

    frontier_vols = []
    valid_targets = []

    for target_return in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, tr=target_return: np.dot(w, mu) - tr}
        )

        # Utilisez une initialisation robuste
        initial_weights = np.ones(len(assets)) / len(assets)

        result = minimize(
            lambda w: np.sqrt(np.dot(w.T, np.dot(Sigma, w))),
            x0=initial_weights,
            method='SLSQP',
            bounds=tuple((0, 1) for _ in assets),
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        if result.success:
            vol = np.sqrt(np.dot(result.x.T, np.dot(Sigma, result.x)))
            frontier_vols.append(vol)
            valid_targets.append(target_return)
        else:
            print(f"Optimisation échouée pour le rendement cible {target_return}")

    if len(frontier_vols) == 0:
        st.warning("Aucun point valide pour la frontière efficace. Vérifiez les données d'entrée.")
        return

    # Tracez la frontière efficace
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier_vols,
        y=valid_targets,
        mode='lines+markers',
        name='Frontière efficace'
    ))

    fig.update_layout(
        title="Frontière efficace du portefeuille",
        xaxis_title="Volatilité annualisée (écart-type)",
        yaxis_title="Rendement annualisé",
        template="plotly_dark"
    )

    # Ajustez les axes pour une meilleure visualisation
    fig.update_xaxes(range=[0, max(frontier_vols) * 1.2])
    fig.update_yaxes(range=[min(valid_targets) * 0.9, max(valid_targets) * 1.1])

    st.plotly_chart(fig, use_container_width=True)

def plot_sector_allocation(allocation):
    sectors = {
        "ENGI.PA": "Énergie", "BNP.PA": "Finance", "ACA.PA": "Finance", "GLE.PA": "Finance",
        "MC.PA": "Luxe", "OR.PA": "Consommation", "AIR.PA": "Aéronautique", "RNO.PA": "Automobile",
        "VK.PA": "Industrie", "KER.PA": "Luxe", "RMS.PA": "Luxe", "SAF.PA": "Aéronautique",
        "HO.PA": "Défense", "SU.PA": "Industrie", "CAP.PA": "Technologie", "STMPA.PA": "Technologie",
        "EDF.PA": "Énergie", "VIE.PA": "Environnement", "EN.PA": "Construction", "SAN.PA": "Santé",
        "SGO.PA": "Matériaux", "ORA.PA": "Télécom", "CA.PA": "Distribution", "RI.PA": "Consommation",
        "DG.PA": "Construction", "AI.PA": "Industrie"
    }
    sector_allocation = allocation.copy()
    sector_allocation["Secteur"] = sector_allocation.index.map(sectors)
    sector_allocation["Secteur"] = sector_allocation["Secteur"].fillna("Autre")
    sector_allocation = sector_allocation.groupby("Secteur")["Poids"].sum()
    st.bar_chart(sector_allocation)

def plot_correlation_matrix(returns):
    corr_matrix = returns.corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
        )
    )
    fig.update_layout(
        title="Matrice de corrélation des actifs",
        xaxis_title="Actifs",
        yaxis_title="Actifs",
        width=400,
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
