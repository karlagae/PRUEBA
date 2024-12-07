import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# Configuración inicial
st.title("Análisis de Portafolio con Modelo de Black-Litterman")
st.sidebar.header("Parámetros del Portafolio")

# Selección de tickers y fechas
tickers = st.sidebar.text_input("Ingrese los tickers separados por comas", "AAPL, MSFT, GOOGL, AMZN")
tickers_list = [t.strip() for t in tickers.split(",")]

start_date = st.sidebar.date_input("Fecha de inicio", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("Fecha de fin", pd.to_datetime("2023-01-01"))

# Descarga de datos
@st.cache
def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    returns = data.pct_change().dropna()
    return data, returns

data, returns = download_data(tickers_list, start_date, end_date)

st.subheader("Precios históricos")
st.line_chart(data)

# Parámetros para Black-Litterman
st.sidebar.header("Parámetros del Modelo Black-Litterman")
risk_free_rate = st.sidebar.number_input("Tasa libre de riesgo (%)", value=2.0) / 100
tau = st.sidebar.number_input("Tau (escala de incertidumbre)", value=0.05)
P = st.sidebar.text_area("Matriz P (opciones del inversor)", "1 0 -1\n0 1 -1")
Q = st.sidebar.text_area("Matriz Q (retornos esperados, en %)", "0.02\n0.01")

# Matrices P y Q
try:
    P_matrix = np.array([list(map(float, row.split())) for row in P.split("\n") if row])
    Q_vector = np.array(list(map(float, Q.split()))) / 100
except ValueError:
    st.error("Error en las matrices P o Q. Por favor, revisa el formato.")
    P_matrix, Q_vector = None, None

# Cálculo de los pesos usando Black-Litterman
if P_matrix is not None and Q_vector is not None:
    market_cap = np.ones(len(tickers_list)) / len(tickers_list)  # Capitalización igualitaria
    cov_matrix = returns.cov().values
    pi = market_cap @ cov_matrix  # Retornos implícitos del mercado

    # Modelo Black-Litterman
    M_inverse = np.linalg.inv(tau * cov_matrix)
    Omega_inverse = np.linalg.inv(np.diag(np.full(P_matrix.shape[0], tau)))

    # Asegurarse de que P_matrix tenga las dimensiones correctas
    # Verifica que el número de activos coincida con las dimensiones de P_matrix
    if P_matrix.shape[1] != len(tickers_list):
        st.error("La matriz P no tiene el número adecuado de columnas para coincidir con los activos.")
    else:
        try:
            # Cálculo de la matriz combinada en el modelo de Black-Litterman
            combined_matrix = M_inverse + P_matrix.T @ Omega_inverse @ P_matrix
            if np.linalg.det(combined_matrix) == 0:
                st.error("La matriz combinada no es invertible, su determinante es 0.")
            else:
                combined_cov = np.linalg.inv(combined_matrix)
                combined_returns = combined_cov @ (M_inverse @ pi + P_matrix.T @ Omega_inverse @ Q_vector)

                # Pesos óptimos
                weights_bl = np.linalg.solve(cov_matrix, combined_returns)

                # Normalización de pesos
                weights_bl /= weights_bl.sum()

                # Mostrar resultados
                st.subheader("Pesos del Portafolio con Black-Litterman")
                weights_df = pd.DataFrame({
                    "Ticker": tickers_list,
                    "Peso": weights_bl
                })
                st.table(weights_df)

                # Gráfico de pesos
                fig = px.pie(weights_df, values="Peso", names="Ticker", title="Distribución de Pesos")
                st.plotly_chart(fig)

        except np.linalg.LinAlgError as e:
            st.error(f"Error al calcular la matriz combinada o sus operaciones: {str(e)}")
else:
    st.warning("Por favor, revisa las matrices P y Q para continuar.")

# Visualización adicional
st.subheader("Riesgo y Retorno del Portafolio")
expected_returns = returns.mean()
portfolio_return = weights_bl @ expected_returns
portfolio_risk = np.sqrt(weights_bl @ cov_matrix @ weights_bl.T)

st.write(f"Dimensiones de M_inverse: {M_inverse.shape}")
st.write(f"Dimensiones de P_matrix: {P_matrix.shape}")
st.write(f"Dimensiones de Omega_inverse: {Omega_inverse.shape}")
st.write(f"**Retorno esperado:** {portfolio_return:.2%}")
st.write(f"**Riesgo (desviación estándar):** {portfolio_risk:.2%}")

# Añadir la visualización de la asignación óptima
st.subheader("Asignación Óptima de Activos")
st.table(weights_df)

# Gráfico de la asignación de activos con el portafolio óptimo
fig_pie_optimal = px.pie(weights_df, names='Ticker', values='Peso', title="Asignación Óptima de Activos (Modelo Black-Litterman)")
st.plotly_chart(fig_pie_optimal)
