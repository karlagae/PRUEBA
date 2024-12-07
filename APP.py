import numpy as np
import scipy.optimize as sco
import yfinance as yf
import pandas as pd
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

# Función para calcular estadísticas del portafolio
def portfolio_stats(weights):
    weights = np.array(weights)[:, np.newaxis]  # Asegura que los pesos estén en una columna
    port_rets = weights.T @ np.array(returns.mean() * 252)[:, np.newaxis]  # Rendimiento esperado anualizado
    port_vols = np.sqrt(np.dot(np.dot(weights.T, returns.cov() * 252), weights))  # Volatilidad anualizada
    return np.array([port_rets, port_vols, port_rets / port_vols]).flatten()  # Retorno, volatilidad y Sharpe ratio

# Función para la optimización del máximo Sharpe Ratio
def min_sharpe_ratio(weights):
    return -portfolio_stats(weights)[2]  # Maximizar el Sharpe ratio

# Restricciones y límites para la optimización
bnds = tuple((0, 1) for x in range(numofasset))
cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})  # Los pesos deben sumar 1
initial_wts = numofasset * [1. / numofasset]  # Inicializar con pesos iguales

# Optimización para el máximo Sharpe Ratio
opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

# Obtener pesos del portafolio con máximo Sharpe ratio
max_sharpe_wts = opt_sharpe['x']

# Optimización para la mínima volatilidad
def min_variance(weights):
    return portfolio_stats(weights)[1]**2  # Minimizar la varianza (volatilidad al cuadrado)

# Optimización para mínima volatilidad
opt_var = sco.minimize(min_variance, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

# Obtener pesos del portafolio con mínima volatilidad
min_volatility_wts = opt_var['x']

# Evaluación de portafolios con los datos de 2021 a 2023
# Rendimiento acumulado y comparación con el S&P 500
all_symbols = symbols + ['^GSPC']  # Incluir el S&P 500
df_all = download_data(all_symbols, start_date='2021-01-01', end_date='2023-12-31')

# Calcular rendimientos diarios
returns_all = df_all.pct_change().fillna(0)

# Calcular rendimientos acumulados para cada portafolio
cumulative_returns = (returns_all + 1).cumprod() - 1

# Calcular métricas como sesgo, curtosis, VaR, CVaR, y otros
def portfolio_metrics(returns):
    # Rendimiento anualizado
    annualized_return = np.mean(returns) * 252
    # Volatilidad anualizada
    annualized_volatility = np.std(returns) * np.sqrt(252)
    # Sesgo
    skewness = stats.skew(returns)
    # Exceso de curtosis
    kurtosis = stats.kurtosis(returns)
    # VaR al 95%
    var_95 = np.percentile(returns, 5)
    # CVaR al 95%
    cvar_95 = returns[returns <= var_95].mean()
    # Sharpe Ratio
    sharpe_ratio = annualized_return / annualized_volatility
    # Sortino Ratio
    downside_returns = returns[returns < 0]
    sortino_ratio = annualized_return / np.std(downside_returns) if len(downside_returns) > 0 else np.nan
    # Drawdown
    cumulative_returns = (returns + 1).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    return {
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'VaR 95%': var_95,
        'CVaR 95%': cvar_95,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown
    }

# Obtener las métricas para cada portafolio
metrics_max_sharpe = portfolio_metrics(cumulative_returns[symbols].dot(max_sharpe_wts))
metrics_min_volatility = portfolio_metrics(cumulative_returns[symbols].dot(min_volatility_wts))
metrics_equal_weight = portfolio_metrics(cumulative_returns[symbols].dot(np.ones(numofasset) / numofasset))
metrics_sp500 = portfolio_metrics(cumulative_returns['^GSPC'])

# Organizar las métricas en un DataFrame
metrics_df = pd.DataFrame({
    'Portfolio': ['Max Sharpe', 'Min Volatility', 'Equal Weight', 'S&P 500'],
    'Annualized Return': [metrics_max_sharpe['Annualized Return'], metrics_min_volatility['Annualized Return'],
                         metrics_equal_weight['Annualized Return'], metrics_sp500['Annualized Return']],
    'Annualized Volatility': [metrics_max_sharpe['Annualized Volatility'], metrics_min_volatility['Annualized Volatility'],
                             metrics_equal_weight['Annualized Volatility'], metrics_sp500['Annualized Volatility']],
    'Sharpe Ratio': [metrics_max_sharpe['Sharpe Ratio'], metrics_min_volatility['Sharpe Ratio'],
                     metrics_equal_weight['Sharpe Ratio'], metrics_sp500['Sharpe Ratio']],
    'Sortino Ratio': [metrics_max_sharpe['Sortino Ratio'], metrics_min_volatility['Sortino Ratio'],
                      metrics_equal_weight['Sortino Ratio'], metrics_sp500['Sortino Ratio']],
    'Max Drawdown': [metrics_max_sharpe['Max Drawdown'], metrics_min_volatility['Max Drawdown'],
                      metrics_equal_weight['Max Drawdown'], metrics_sp500['Max Drawdown']],
    'Skewness': [metrics_max_sharpe['Skewness'], metrics_min_volatility['Skewness'],
                 metrics_equal_weight['Skewness'], metrics_sp500['Skewness']],
    'Kurtosis': [metrics_max_sharpe['Kurtosis'], metrics_min_volatility['Kurtosis'],
                 metrics_equal_weight['Kurtosis'], metrics_sp500['Kurtosis']],
    'VaR 95%': [metrics_max_sharpe['VaR 95%'], metrics_min_volatility['VaR 95%'],
                metrics_equal_weight['VaR 95%'], metrics_sp500['VaR 95%']],
    'CVaR 95%': [metrics_max_sharpe['CVaR 95%'], metrics_min_volatility['CVaR 95%'],
                 metrics_equal_weight['CVaR 95%'], metrics_sp500['CVaR 95%']],
})

# Mostrar el DataFrame con las métricas en Streamlit
st.subheader("Métricas de los Portafolios")
st.dataframe(metrics_df)

# Graficar el rendimiento acumulado de los portafolios
fig = go.Figure()

# Portafolio con máximo Sharpe
fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns
