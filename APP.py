import numpy as np
import pandas as pd
from scipy.linalg import inv
import yfinance as yf

def black_litterman_prior_equilibrium(returns, market_caps, risk_aversion):
    """
    Cálculo del rendimiento esperado de equilibrio basado en los retornos históricos.
    """
    market_weights = market_caps / np.sum(market_caps)
    equilibrium_returns = risk_aversion * np.dot(returns.cov(), market_weights)
    return equilibrium_returns

def black_litterman_model(returns, market_caps, risk_aversion, P, Q, omega=None):
    """
    Modelo Black-Litterman.
    """
    # Rendimientos esperados de equilibrio
    pi = black_litterman_prior_equilibrium(returns, market_caps, risk_aversion)
    
    # Matriz de covarianza del mercado
    sigma = returns.cov()

    # Si no se especifica omega, se asume proporcional a la varianza de las vistas
    if omega is None:
        omega = np.diag(np.diag(P @ sigma @ P.T))
    
    # Black-Litterman posterior
    tau = 0.025  # Escala de incertidumbre
    posterior_returns = inv(inv(tau * sigma) + P.T @ inv(omega) @ P) @ (inv(tau * sigma) @ pi + P.T @ inv(omega) @ Q)
    posterior_covariance = inv(inv(tau * sigma) + P.T @ inv(omega) @ P)
    
    return posterior_returns, posterior_covariance
# Descargar datos históricos
symbols = ['LQD', 'EMB', 'VTI', 'EEM', 'GLD']
df = yf.download(symbols, start='2010-01-01', end='2023-12-31')['Close']
returns = df.pct_change().dropna()

# Supongamos capitalizaciones de mercado aproximadas
market_caps = np.array([30e9, 20e9, 50e9, 10e9, 40e9])  # Ajusta según datos reales

# Parámetros iniciales
risk_aversion = 3  # Aversión al riesgo promedio del mercado
P = np.array([
    [1, -1, 0, 0, 0],  # Vista 1: LQD > EMB
    [0, 0, 1, 0, -1]   # Vista 2: VTI > GLD
])
Q = np.array([0.02, 0.01])  # Suposiciones del inversionista (rendimientos esperados)

# Aplicar Black-Litterman
bl_returns, bl_covariance = black_litterman_model(returns, market_caps, risk_aversion, P, Q)

# Optimización del portafolio con Black-Litterman
from scipy.optimize import minimize

def optimize_portfolio(expected_returns, covariance_matrix):
    num_assets = len(expected_returns)
    initial_weights = np.ones(num_assets) / num_assets

    def portfolio_volatility(weights):
        return np.sqrt(weights.T @ covariance_matrix @ weights)

    def constraint(weights):
        return np.sum(weights) - 1

    bounds = [(0, 1) for _ in range(num_assets)]
    constraints = {'type': 'eq', 'fun': constraint}

    result = minimize(
        portfolio_volatility,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x

# Optimizar portafolio
optimal_weights = optimize_portfolio(bl_returns, bl_covariance)
optimized_portfolio = dict(zip(symbols, np.round(optimal_weights, 4)))

# Mostrar resultados
print("Pesos optimizados Black-Litterman:")
print(optimized_portfolio)

st.title("Optimización del Portafolio con Black-Litterman")
st.subheader("Pesos del portafolio")
st.write(optimized_portfolio)

# Rendimiento esperado y volatilidad del portafolio optimizado
portfolio_return = np.dot(optimal_weights, bl_returns)
portfolio_volatility = np.sqrt(optimal_weights.T @ bl_covariance @ optimal_weights)
st.write(f"Rendimiento esperado: {portfolio_return:.2%}")
st.write(f"Volatilidad esperada: {portfolio_volatility:.2%}")
