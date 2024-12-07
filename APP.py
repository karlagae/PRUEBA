import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

# Definir los activos y el período
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Ejemplo de activos
start_date = '2020-01-01'
end_date = '2024-01-01'

# Descargar los datos de precios históricos de Yahoo Finance
data = yf.download(assets, start=start_date, end=end_date)['Adj Close']

# Calcular los rendimientos logarítmicos
returns = np.log(data / data.shift(1)).dropna()

# Calcular la matriz de covarianza de los rendimientos
market_covariance = returns.cov().values

# Definir la matriz de incertidumbre M (puedes ajustarla según tu análisis)
M = np.diag([0.02] * len(assets))  # Ejemplo de incertidumbre igual para cada activo

# Definir la matriz de vistas P (cuantifica las creencias del inversor)
# Ejemplo: creemos que AAPL tendrá un rendimiento superior al de AMZN
P = np.array([[1, 0, 0, 0, -1]])  # 1 para AAPL, -1 para AMZN

# Vector de las vistas Q (nuestras creencias sobre los rendimientos)
Q = np.array([0.05])  # Creemos que la diferencia entre AAPL y AMZN será del 5%

# Regularización para evitar singularidad en las matrices
epsilon = 1e-6  # Pequeña constante para regularizar las matrices
market_covariance += np.eye(len(market_covariance)) * epsilon  # Regularizar la covarianza
M += np.eye(len(M)) * epsilon  # Regularizar la matriz de incertidumbre

# Calcular el término medio (Black-Litterman)
middle_term = np.linalg.inv(np.dot(np.dot(P.T, np.linalg.inv(M)), P) + np.linalg.inv(market_covariance))

# Calcular la solución final usando el modelo Black-Litterman
BL_result = np.dot(np.dot(np.dot(np.linalg.inv(market_covariance), P.T), np.linalg.inv(M)), Q) + np.dot(np.dot(np.linalg.inv(market_covariance), np.ones(len(assets))), np.ones(len(assets)))

# Mostrar resultados
st.title("Modelo Black-Litterman")
st.write("Rendimientos esperados ajustados según las vistas del inversor:")
st.write(BL_result)

# Mostrar gráfico de los activos y su covarianza
st.subheader("Covarianza de los rendimientos")
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(market_covariance, cmap='coolwarm', interpolation='none')
ax.set_title("Covarianza entre los activos")
ax.set_xticks(np.arange(len(assets)))
ax.set_yticks(np.arange(len(assets)))
ax.set_xticklabels(assets)
ax.set_yticklabels(assets)
plt.colorbar(ax.imshow(market_covariance, cmap='coolwarm', interpolation='none'))
st.pyplot(fig)
