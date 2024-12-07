#---------------------------------------------------------------------------------------------------#
#                                      CARGA DE LIBRERIAS  

import pandas as pd
import numpy as np
from numpy.linalg import multi_dot
import scipy.optimize as sco
import yfinance as yf  # Importamos yfinance para obtener datos de Yahoo Finance
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import datetime as dt

#---------------------------------------------------------------------------------------------------#
#                                             PAGE INFO

st.set_page_config(
    page_title="Portfolio Optimization",
    page_icon="mag"
)
st.title("Portfolio Optimization & Backtesting")

#---------------------------------------------------------------------------------------------------#
# CARGA DE DATOS DE YAHOO FINANCE

# Definimos los tickers que queremos analizar
tickers = st.text_input("Introduce los tickers separados por comas (ej. AAPL, MSFT, GOOGL):", "AAPL, MSFT, GOOGL")

# Definimos el rango de fechas
start_date = st.date_input("Fecha de inicio:", value=dt.date(2021, 1, 1))
end_date = st.date_input("Fecha de fin:", value=dt.date(2023, 1, 1))

if st.button("Cargar datos"):
    # Cargamos los datos de Yahoo Finance
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    st.session_state.data = data
    st.session_state.returns = data.pct_change().dropna()
    st.success("Datos cargados exitosamente.")
    st.write("Precios de cierre ajustados:")
    st.dataframe(data.tail())
    st.write("Retornos diarios:")
    st.dataframe(st.session_state.returns.tail())
else:
    st.warning("Por favor, introduce los tickers y selecciona las fechas para cargar los datos.")

#---------------------------------------------------------------------------------------------------#
#                                 OPTIMIZACION DE PORTAFOLIOS

# Definimos la función portfolio stats para calcular retornos, volatilidad y Sharpe ratio de los portafolios
def portfolio_stats(weights, returns, return_df=False):
    weights = np.array(weights)[:, np.newaxis]
    port_rets = weights.T @ np.array(returns.mean() * 252)[:, np.newaxis]
    port_vols = np.sqrt(multi_dot([weights.T, returns.cov() * 252, weights]))
    sharpe_ratio = port_rets / port_vols
    resultados = np.array([port_rets, port_vols, sharpe_ratio]).flatten()
    
    if return_df:
        return pd.DataFrame(data=np.round(resultados, 4),
                            index=["Returns", "Volatility", "Sharpe_Ratio"],
                            columns=["Resultado"])
    else:
        return resultados

st.markdown("## Optimization :muscle:")

# Definimos las fechas sobre las que queremos optimizar el portafolio
opt_range = st.slider("Selecciona un rango de fechas:", min_value=start_date, 
                      max_value=end_date, value=(start_date, end_date),
                      format="YYYY-MM-DD") 
st.session_state.start_date_opt, st.session_state.end_date_opt = opt_range

st.write("Inicio:", st.session_state.start_date_opt)
st.write("Fin:", st.session_state.end_date_opt)

if "returns1" not in st.session_state:
    st.session_state.returns1 = None

# Guardamos los retornos en un nuevo df
if st.session_state.returns is not None:
    st.session_state.returns1 = st.session_state.returns.loc[st.session_state.start_date_opt:st.session_state.end_date_opt]

# ingresar el rendimiento objetivo del portafolio de mínima varianza con rendimiento objetivo
r_obj = st.number_input(
    "Especifica el rendimiento objetivo para el portafolio de mínima volatilidad:",
    value=0.1, min_value=0.0, max_value=1.0
)

opt_bool = False
if st.button("¡Vamos!"):
    opt_bool = True

# Definimos la función que nos ayudará a obtener la volatilidad del portafolio
def get_volatility(weights, returns):
    return portfolio_stats(weights, returns)[1]

# Función para optimizar el portafolio bajo mínima volatilidad
def min_vol_opt(returns):
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})  # Restricción de que la suma de los pesos sea 1
    bnds = tuple((0, 1) for x in range(len(returns.columns)))  # Restricción de que los pesos estén entre 0 y 1
        initial_wts = np.array(len(returns.columns) * [1. / len(returns.columns)])  # Pesos iniciales

    # Optimización
    result = sco.minimize(get_volatility, initial_wts, args=(returns,), method='SLSQP', bounds=bnds, constraints=cons)
    return result

if opt_bool and st.session_state.returns1 is not None:
    # Ejecutamos la optimización
    optimal_weights = min_vol_opt(st.session_state.returns1)

    # Mostramos los resultados
    st.write("Pesos óptimos del portafolio:")
    st.write(optimal_weights.x)

    # Calculamos estadísticas del portafolio óptimo
    stats = portfolio_stats(optimal_weights.x, st.session_state.returns1, return_df=True)
    st.write("Estadísticas del portafolio óptimo:")
    st.dataframe(stats)

    # Gráfica de la distribución de pesos
    fig = go.Figure(data=[go.Pie(labels=st.session_state.returns1.columns, values=optimal_weights.x, hole=.3)])
    fig.update_layout(title_text='Distribución de Pesos del Portafolio Óptimo')
    st.plotly_chart(fig)

#---------------------------------------------------------------------------------------------------#
#                                 BACKTESTING DEL PORTAFOLIO

st.markdown("## Backtesting :chart_with_upwards_trend:")

if st.button("Ejecutar Backtest"):
    if st.session_state.data is not None and opt_bool:
        # Simulamos el rendimiento del portafolio
        portfolio_returns = (st.session_state.returns1 * optimal_weights.x).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # Gráfica de rendimiento acumulado
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_returns, label='Rendimiento del Portafolio Óptimo', color='blue')
        plt.title('Rendimiento Acumulado del Portafolio Óptimo')
        plt.xlabel('Fecha')
        plt.ylabel('Rendimiento Acumulado')
        plt.legend()
        plt.grid()
        st.pyplot(plt)

        # Estadísticas del rendimiento del portafolio
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(cumulative_returns)) - 1
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility

        st.write("Estadísticas del Backtest:")
        st.write(f"Rendimiento Total: {total_return:.2%}")
        st.write(f"Rendimiento Anualizado: {annualized_return:.2%}")
        st.write(f"Volatilidad Anualizada: {annualized_volatility:.2%}")
        st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Gráfica de los retornos diarios del portafolio
        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_returns, label='Retornos Diarios del Portafolio Óptimo', color='orange')
        plt.title('Retornos Diarios del Portafolio Óptimo')
        plt.xlabel('Fecha')
        plt.ylabel('Retorno Diario')
        plt.legend()
        plt.grid()
        st.pyplot(plt)
