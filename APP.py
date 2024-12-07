import numpy as np
import pandas as pd
import yfinance as yf
import scipy.optimize as sco
from datetime import date
from scipy import stats
import plotly.graph_objects as go
import streamlit as st

# Símbolos de los ETFs
symbols = ['LQD', 'EMB', 'VTI', 'EEM', 'GLD']
numofasset = len(symbols)  # Número de activos

# Descargar los datos de los ETFs
def download_data(tickers, start_date='2010-01-01', end_date=date.today().strftime('%Y-%m-%d')):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Close']

# Descargar datos de 2021 a 2023
df = download_data(symbols, start_date='2021-01-01', end_date='2023-12-31')

# Calcular rendimientos diarios
returns = df.pct_change().fillna(0)

# Definir el modelo Black-Litterman

# Suponemos las expectativas del mercado como el rendimiento medio de los activos
market_returns = returns.mean() * 252  # Rendimiento anualizado
market_covariance = returns.cov() * 252  # Covarianza anualizada

# Opiniones del inversor: supongamos que el inversor tiene una opinión sobre ciertos activos
# Definir las opiniones (por ejemplo, un rendimiento esperado para VTI de 10% y para EEM de 8%)
opinions = np.array([0.10, 0.08])  # Creencias del inversor sobre VTI y EEM
P = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])  # Matriz de vistas (impacto en los activos VTI y EEM)
Q = np.array([0.10, 0.08])  # Opiniones (rendimientos esperados para VTI y EEM)

# Matriz de incertidumbre sobre las opiniones
tau = 0.025  # Valor de incertidumbre
M = tau * market_covariance  # Matriz de incertidumbre

# Calcular la matriz ajustada (Black-Litterman)
middle_term = np.linalg.inv(np.dot(np.dot(P.T, np.linalg.inv(M)), P) + np.linalg.inv(market_covariance))
adj_returns = np.dot(middle_term, np.dot(np.dot(P.T, np.linalg.inv(M)), Q) + np.dot(np.linalg.inv(market_covariance), market_returns))

# Los rendimientos ajustados ahora contienen las expectativas del mercado y las opiniones del inversor
adjusted_returns = pd.Series(adj_returns, index=symbols)

# Optimización con Black-Litterman

# Función para calcular estadísticas del portafolio
def portfolio_stats(weights, expected_returns, covariance_matrix):
    weights = np.array(weights)[:, np.newaxis]
    port_rets = weights.T @ np.array(expected_returns)[:, np.newaxis]
    port_vols = np.sqrt(np.dot(np.dot(weights.T, covariance_matrix), weights))
    return np.array([port_rets, port_vols, port_rets / port_vols]).flatten()

# Función para la optimización del máximo Sharpe Ratio
def min_sharpe_ratio(weights, expected_returns, covariance_matrix):
    return -portfolio_stats(weights, expected_returns, covariance_matrix)[2]  # Maximizar el Sharpe ratio

# Restricciones y límites para la optimización
bnds = tuple((0, 1) for x in range(numofasset))
cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})  # Los pesos deben sumar 1
initial_wts = numofasset * [1. / numofasset]  # Inicializar con pesos iguales

# Optimización para el máximo Sharpe Ratio con los rendimientos ajustados de Black-Litterman
opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, args=(adjusted_returns, market_covariance), method='SLSQP', bounds=bnds, constraints=cons)

# Obtener los pesos del portafolio con el máximo Sharpe ratio
max_sharpe_wts = opt_sharpe['x']

# Optimización para la mínima volatilidad
def min_variance(weights, expected_returns, covariance_matrix):
    return portfolio_stats(weights, expected_returns, covariance_matrix)[1]**2  # Minimizar la varianza (volatilidad al cuadrado)

# Optimización para mínima volatilidad con los rendimientos ajustados de Black-Litterman
opt_var = sco.minimize(min_variance, initial_wts, args=(adjusted_returns, market_covariance), method='SLSQP', bounds=bnds, constraints=cons)

# Obtener los pesos del portafolio con mínima volatilidad
min_volatility_wts = opt_var['x']

# Visualización y métricas del portafolio
st.title("Optimización de Portafolio con Black-Litterman")

st.subheader("Portafolio con Máximo Sharpe Ratio (ajustado por Black-Litterman)")
st.write(dict(zip(symbols, np.around(max_sharpe_wts, 2))))

st.subheader("Portafolio con Mínima Volatilidad (ajustado por Black-Litterman)")
st.write(dict(zip(symbols, np.around(min_volatility_wts, 2))))

# Graficar la frontera eficiente con los rendimientos ajustados de Black-Litterman
# Se sigue el mismo proceso de frontera eficiente como antes, pero usando `adjusted_returns`

targetrets = np.linspace(0.02, 0.30, 100)
tvols = []

for tr in targetrets:
    ef_cons = ({
        'type': 'eq', 'fun': lambda x: portfolio_stats(x, adjusted_returns, market_covariance)[0] - tr
    }, {
        'type': 'eq', 'fun': lambda x: np.sum(x) - 1
    })
    opt_ef = sco.minimize(min_variance, initial_wts, args=(adjusted_returns, market_covariance), method='SLSQP', bounds=bnds, constraints=ef_cons)
    tvols.append(np.sqrt(opt_ef['fun']))

targetvols = np.array(tvols)

efport = pd.DataFrame({
    'targetrets': np.around(100 * targetrets, 2),
    'targetvols': np.around(100 * targetvols, 2),
    'targetsharpe': np.around(targetrets / targetvols, 2)
})

# Graficar la frontera eficiente con Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=efport['targetvols'], y=efport['targetrets'],
    mode='lines', name='Frontera Eficiente',
    line=dict(color='blue')
))

fig.update_layout(
    title="Frontera Eficiente con Black-Litterman",
    xaxis=dict(title="Volatilidad Esperada (%)"),
    yaxis=dict(title="Rendimiento Esperado (%)")
)

st.plotly_chart(fig)

