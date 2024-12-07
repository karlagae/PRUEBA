import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import scipy.optimize as sco
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go





# Descargar los datos del S&P 500 desde Yahoo Finance
@st.cache_data
def load_benchmark_data(symbol, start_date='2000-01-01'):
    data = yf.download(symbol, start=start_date)
    return data

# Función para calcular estadísticas del portafolio
def portfolio_stats(weights, returns, return_df=False):
    weights = np.array(weights)
    portfolio_return = np.dot(weights, returns.mean())
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility

    if return_df:
        data = {
            "Resultado": [portfolio_return, portfolio_volatility, sharpe_ratio],
            "Métrica": ["Retorno", "Volatilidad", "Ratio de Sharpe"]
        }
        return pd.DataFrame(data).set_index("Métrica")
    else:
        return portfolio_return, portfolio_volatility, sharpe_ratio

# Función para obtener la volatilidad
def get_volatility(weights, returns):
    return portfolio_stats(weights, returns)[1]

# Optimización de Sharpe Ratio Máximo
def max_sr_opt(returns):
    cons = {'type': 'eq', 'fun': lambda x: sum(x) - 1}
    bnds = tuple((0, 1) for _ in range(len(returns.columns)))
    initial_wts = np.array(len(returns.columns) * [1. / len(returns.columns)])
    
    opt_sr = sco.minimize(fun=lambda x: -portfolio_stats(x, returns)[2], 
                          x0=initial_wts, bounds=bnds, constraints=cons)
    
    max_sr_pesos = pd.DataFrame(data=np.around(opt_sr['x'] * 100, 2),
                                index=returns.columns, 
                                columns=["Max_SR"])
    max_sr_stats = portfolio_stats(opt_sr['x'], returns, return_df=True)
    max_sr_stats = max_sr_stats.rename(columns={"Resultado": "Max_SR"})
    
    return {"max_sr_pesos": max_sr_pesos, "max_sr_stats": max_sr_stats}

# Optimización de Mínima Volatilidad con Retorno Objetivo
def min_vol_obj_opt(returns, r_obj):
    cons = ({'type': 'eq', 'fun': lambda x: portfolio_stats(x, returns)[0] - r_obj},
            {'type': 'eq', 'fun': lambda x: sum(x) - 1})
    bnds = tuple((0, 1) for _ in range(len(returns.columns)))
    initial_wts = np.array(len(returns.columns) * [1. / len(returns.columns)])
    
    opt_min_obj = sco.minimize(fun=get_volatility, x0=initial_wts, args=(returns), 
                               method='SLSQP', bounds=bnds, constraints=cons)
    
    min_obj_pesos = pd.DataFrame(data=np.around(opt_min_obj['x'] * 100, 2),
                                 index=returns.columns, 
                                 columns=["Min_Vol_Obj"])
    min_obj_stats = portfolio_stats(opt_min_obj['x'], returns, return_df=True)
    min_obj_stats = min_obj_stats.rename(columns={"Resultado": "Min_Vol_Obj"})
    
    return {"min_obj_pesos": min_obj_pesos, "min_obj_stats": min_obj_stats}

# Variables globales para los resultados de optimización
if "max_sr_resultados" not in st.session_state:
    st.session_state.max_sr_resultados = None
if "min_obj_resultados" not in st.session_state:
    st.session_state.min_obj_resultados = None
opt_bool = False  # Valor predeterminado

import streamlit as st

# Ejemplo de control con un checkbox
opt_bool = st.checkbox("¿Ejecutar optimización?", value=False)

# O si depende de otro control (por ejemplo, un botón)
if st.button("Ejecutar optimización"):
    opt_bool = True

# Verificar que `opt_bool` y los datos estén definidos antes de optimizar
if opt_bool:
    # Verifica si los datos necesarios están disponibles (ej. 'returns1' en session_state)
    if st.session_state.get('returns1') is not None:
        try:
            # Optimización de Maximum Sharpe Ratio (por ejemplo)
            st.session_state.max_sr_resultados = max_sr_opt(st.session_state.returns1)
            st.success("Maximum Sharpe Ratio Portfolio successfully optimized!")
        except Exception as e:
            st.warning(f"An error occurred while optimizing Maximum Sharpe Ratio: {e}")

        try:
            # Optimización de Minimum Volatility Portfolio
            st.session_state.min_obj_resultados = min_vol_obj_opt(st.session_state.returns1, r_obj)
            st.success("Minimum Volatility Portfolio with Target Return successfully optimized!")
        except Exception as e:
            st.warning(f"An error occurred while optimizing Minimum Volatility Portfolio: {e}")
    else:
        st.warning("No data found for optimization!")


# Mostrar resultados de optimización
if st.session_state.max_sr_resultados is not None:
    st.subheader("Maximum Sharpe Ratio Portfolio")
    st.dataframe(st.session_state.max_sr_resultados["max_sr_pesos"])
    st.dataframe(st.session_state.max_sr_resultados["max_sr_stats"])

if st.session_state.min_obj_resultados is not None:
    st.subheader("Minimum Volatility with Target Return Portfolio")
    st.dataframe(st.session_state.min_obj_resultados["min_obj_pesos"])
    st.dataframe(st.session_state.min_obj_resultados["min_obj_stats"])

# Comparar pesos y métricas entre portafolios
if st.session_state.min_obj_resultados is not None and st.session_state.max_sr_resultados is not None:
    st.subheader("Portfolio Weights Comparison")
    try:
        resultados_pesos = st.session_state.max_sr_resultados["max_sr_pesos"].merge(
            st.session_state.min_obj_resultados["min_obj_pesos"], left_index=True, right_index=True
        )
        st.session_state.resultados_pesos = resultados_pesos

        # Graficar comparación de pesos
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, (col, title) in zip(axes, resultados_pesos.iteritems()):
            ax.pie(col, labels=col.index, autopct='%1.1f%%')
            ax.set_title(title)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error comparing weights: {e}")

# Incorporar Backtesting
# Similar lógica adaptada a los resultados previos...
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Incorporar Backtesting
def backtest_portfolio(weights, returns, initial_capital=10000):
    """
    Realiza el backtesting de un portafolio dado un conjunto de pesos, retornos históricos y capital inicial.
    """
    weights = np.array(weights)
    # Simulando el rendimiento del portafolio multiplicando los pesos por los retornos de los activos
    port_returns = np.dot(returns, weights)
    
    # Calculando el valor del portafolio con el capital inicial
    portfolio_value = initial_capital * (1 + port_returns).cumprod()
    
    return portfolio_value

# Función para comparar el rendimiento de diferentes portafolios con un benchmark
def plot_backtest_comparison(returns, weights_max_sr, weights_min_vol_obj, benchmark_returns, initial_capital=10000):
    """
    Compara el rendimiento de los portafolios (Sharpe Ratio y Minimum Volatility) con un benchmark.
    """
    # Realizando el backtest para el portafolio con máximo Sharpe Ratio
    portfolio_max_sr = backtest_portfolio(weights_max_sr, returns, initial_capital)
    
    # Realizando el backtest para el portafolio con mínima volatilidad
    portfolio_min_vol_obj = backtest_portfolio(weights_min_vol_obj, returns, initial_capital)
    
    # Realizando el backtest para el benchmark
    benchmark_value = initial_capital * (1 + benchmark_returns).cumprod()

    # Graficar los resultados
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_max_sr, label='Portfolio Max Sharpe Ratio', color='blue')
    plt.plot(portfolio_min_vol_obj, label='Portfolio Min Volatility', color='green')
    plt.plot(benchmark_value, label='Benchmark', color='black', linestyle='--')
    
    plt.title('Backtest Comparison')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del Portafolio')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Mostrar el gráfico en Streamlit
    st.pyplot(plt)

# Cargar los datos históricos
if 'benchmark_data' not in st.session_state:
    # Suponiendo que el archivo 'sp500.csv' contiene los datos del benchmark
    benchmark_data = pd.read_csv('sp500.csv', index_col='Date', parse_dates=True)
    benchmark_returns = benchmark_data['Adj Close'].pct_change().dropna()
    st.session_state.benchmark_data = benchmark_returns

if 'returns1' not in st.session_state:
    # Suponiendo que el archivo 'returns.csv' contiene los retornos de los activos
    returns1 = pd.read_csv('returns.csv', index_col='Date', parse_dates=True)
    st.session_state.returns1 = returns1.pct_change().dropna()

# Verifica si los resultados de la optimización están disponibles en la sesión
if 'max_sr_resultados' in st.session_state and 'min_obj_resultados' in st.session_state:
    if st.session_state.max_sr_resultados is not None and st.session_state.min_obj_resultados is not None:
        # Extraer los pesos optimizados de las soluciones
        weights_max_sr = st.session_state.max_sr_resultados["max_sr_pesos"].values.flatten()
        weights_min_vol_obj = st.session_state.min_obj_resultados["min_obj_pesos"].values.flatten()

        # Obtener los retornos históricos del portafolio optimizado
        portfolio_returns = st.session_state.returns1
        
        # Obtener los retornos históricos del benchmark
        benchmark_returns = st.session_state.benchmark_data

        # Ejecutar y graficar el backtesting de ambos portafolios comparados con el benchmark
        plot_backtest_comparison(portfolio_returns, weights_max_sr, weights_min_vol_obj, benchmark_returns)
