import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import streamlit as st
from streamlit_option_menu import option_menu  # Para un menú de navegación atractivo

# Configuración inicial de la página
st.set_page_config(
    page_title="Análisis de ETFs",
    page_icon="📈",
    layout="wide"
)

# Encabezado y descripción
st.title("📊 Análisis Avanzado de ETFs")
st.markdown("""
Explora y analiza las principales métricas estadísticas de los ETFs seleccionados.
Optimiza tu portafolio con Sharpe Ratio o mínima volatilidad y visualiza la frontera eficiente.
""")

# Sidebar con menú de navegación
with st.sidebar:
    selected = option_menu(
        "Menú",
        ["Introducción", "Análisis Estadístico", "Optimización de Portafolios"],
        icons=["house", "bar-chart-line", "gear"],
        menu_icon="menu-up",
        default_index=0
    )

# Función para obtener datos históricos
@st.cache
def obtener_datos_acciones(simbolos, start_date, end_date):
    data = yf.download(simbolos, start=start_date, end=end_date)['Close']
    return data.ffill().dropna()

# Funciones para cálculos
def calcular_metricas(df):
    returns = df.pct_change().dropna()
    media = returns.mean()
    sesgo = returns.skew()
    curtosis = returns.kurtosis()
    return returns, media, sesgo, curtosis

def calcular_var_cvar(returns, confidence=0.95):
    VaR = returns.quantile(1 - confidence)
    CVaR = returns[returns <= VaR].mean()
    return VaR, CVaR

def calcular_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod() - 1
    drawdown = cumulative_returns - cumulative_returns.cummax()
    return drawdown.min()

def calcular_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calcular_sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < target_return]
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    return (np.sqrt(252) * excess_returns.mean() / downside_deviation
            if downside_deviation != 0 else np.nan)

# Lógica según la selección del menú
if selected == "Introducción":
    st.header("Bienvenido")
    st.write("""
    Esta herramienta utiliza datos históricos para calcular métricas avanzadas de ETFs, optimizar portafolios,
    y simular la frontera eficiente.
    """)

elif selected == "Análisis Estadístico":
    st.header("🔍 Análisis Estadístico de ETFs")

    etfs = ["LQD", "EMB", "VTI", "EEM", "GLD"]
    selected_etfs = st.multiselect("Selecciona los ETFs para analizar", etfs, default=etfs)
    start_date = st.date_input("Fecha de inicio", datetime(2010, 1, 1))
    end_date = st.date_input("Fecha de fin", datetime(2023, 12, 31))

    if st.button("Calcular métricas"):
        data = obtener_datos_acciones(selected_etfs, start_date, end_date)
        df_resultados = pd.DataFrame()

        for etf in selected_etfs:
            returns, media, sesgo, curtosis = calcular_metricas(data[etf])
            var_95, cvar_95 = calcular_var_cvar(returns)
            drawdown = calcular_drawdown(returns)
            sharpe = calcular_sharpe_ratio(returns)
            sortino = calcular_sortino_ratio(returns)
            df_resultados[etf] = {
                "Media": media,
                "Sesgo": sesgo,
                "Curtosis": curtosis,
                "VaR 95%": var_95,
                "CVaR 95%": cvar_95,
                "Drawdown": drawdown,
                "Sharpe Ratio": sharpe,
                "Sortino Ratio": sortino
            }
        st.subheader("📋 Resultados:")
        st.dataframe(df_resultados.T.style.format("{:.2%}"))

elif selected == "Optimización de Portafolios":
    st.header("⚙️ Optimización de Portafolios")

    st.write("Simula portafolios, maximiza Sharpe Ratio o minimiza volatilidad, y visualiza la Frontera Eficiente.")
    # Aquí incluirás las funciones de optimización y la gráfica de la frontera eficiente.

