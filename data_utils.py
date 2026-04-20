import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data
def get_data(tickers, start_date):
    try:
        data = yf.download(
            tickers,
            start=start_date,
            auto_adjust=True,
            progress=False
        )

        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.levels[0]:
                data = data["Close"]
            elif "Adj Close" in data.columns.levels[0]:
                data = data["Adj Close"]

        # Vérifiez que les données ne sont pas vides
        if data.empty:
            st.error("Aucune donnée téléchargée. Vérifiez les tickers ou la connexion Internet.")
            return pd.DataFrame()

        # Supprimez les colonnes avec des données manquantes
        data = data.dropna(axis=1, how="all")

        # Vérifiez qu'il reste des données après le nettoyage
        if data.empty:
            st.error("Aucune donnée valide après nettoyage.")
            return pd.DataFrame()

        return data

    except Exception as e:
        st.error(f"Erreur lors du téléchargement des données : {e}")
        return pd.DataFrame()
