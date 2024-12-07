import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import streamlit as st

# Funciones auxiliares

def obtener_datos_acciones(simbolos, start_date, end_date):
    """
    Descarga los precios históricos de los símbolos seleccionados entre dos fechas.
    """
    data = yf.download(simbolos, start=start_date, end=end_date)['Close']
    return data.ffill().dropna()

def calcular_metricas(df):
    """
    Calcula métricas estadísticas clave como media, sesgo y curtosis.
    """
    returns = df.pct_change().dropna()
    return {
        "Rendimientos diarios": returns,
        "Media": returns.mean(),
        "Sesgo": returns.skew(),
        "Curtosis": returns.kurtosis()
    }

# Cálculos avanzados
def calcular_var_cvar(returns, confidence=0.95):
    VaR = returns.quantile(1 - confidence)
    CVaR = returns[returns <= VaR].mean()
    return VaR, CVaR

def calcular_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    return max_drawdown

def calcular_sharpe_ratio(returns, risk_free_rate=0.02):
    return np.sqrt(252) * (returns.mean() - risk_free_rate / 252) / returns.std()

def calcular_sortino_ratio(returns, target_return=0, risk_free_rate=0.02):
    downside_returns = returns[returns < target_return]
    downside_deviation = np.sqrt((downside_returns**2).mean())
    return (returns.mean() - risk_free_rate / 252) / downside_deviation

# Configuración de la aplicación
st.set_page_config(page_title="Análisis de ETFs", layout="wide")

# Título y descripción
st.title("📈 Análisis de ETFs")
st.markdown(
    """
    Explora métricas clave y analiza la rentabilidad de ETFs utilizando técnicas avanzadas.
    """
)

# Panel de configuración
st.sidebar.header("Configuración del análisis")
etfs = ["LQD", "EMB", "VTI", "EEM", "GLD"]
selected_etfs = st.sidebar.multiselect("Selecciona los ETFs para analizar:", etfs, default=etfs)

start_date = st.sidebar.date_input("Fecha de inicio:", datetime(2010, 1, 1))
end_date = st.sidebar.date_input("Fecha de fin:", datetime(2023, 12, 31))

# Ejecutar análisis con un botón estilizado
if st.sidebar.button("📊 Analizar ETFs"):
    data = obtener_datos_acciones(selected_etfs, start_date, end_date)
    resultados = {
        etf: {
            "Rendimientos": data[etf].pct_change().dropna(),
            "VaR 95%": calcular_var_cvar(data[etf].pct_change().dropna())[0],
            "Drawdown": calcular_drawdown(data[etf].pct_change().dropna()),
            "Sharpe Ratio": calcular_sharpe_ratio(data[etf].pct_change().dropna())
        }
        for etf in selected_etfs
    }

    # Visualización de métricas
    st.subheader("📊 Métricas principales")
    df_resultados = pd.DataFrame(resultados).T
    st.dataframe(
        df_resultados.style.format({
            "Media": "{:.2%}",
            "VaR 95%": "{:.2%}",
            "Drawdown": "{:.2%}",
            "Sharpe Ratio": "{:.2f}"
        })
    )

    # Gráficos interactivos
    st.subheader("📈 Gráficos de precios históricos")
    fig = go.Figure()
    for etf in selected_etfs:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[etf],
                mode="lines",
                name=etf
            )
        )

    fig.update_layout(
        title="Precios históricos de los ETFs seleccionados",
        xaxis_title="Fecha",
        yaxis_title="Precio de Cierre",
        template="plotly_white"
    )
    st.plotly_chart(fig)
    
    st.subheader("🔄 Rendimientos diarios")
    df_rendimientos = data.pct_change().dropna()
    st.line_chart(df_rendimientos)
