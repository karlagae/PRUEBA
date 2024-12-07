import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import streamlit as st

# Funciones auxiliares

# Función para obtener los datos históricos de precios
def obtener_datos_acciones(simbolos, start_date, end_date):
    data = yf.download(simbolos, start=start_date, end=end_date)['Close']
    return data.ffill().dropna()

# Funciones para calcular métricas estadísticas

def calcular_metricas(df):
    returns = df.pct_change().dropna()  # Calculamos los rendimientos diarios
    media = returns.mean()  # Media
    sesgo = returns.skew()  # Sesgo
    curtosis = returns.kurtosis()  # Curtosis
    return returns, media, sesgo, curtosis

# Métricas de riesgo (VaR y CVaR)
def calcular_var_cvar(returns, confidence=0.95):
    VaR = returns.quantile(1 - confidence)  # Value at Risk
    CVaR = returns[returns <= VaR].mean()  # Conditional VaR
    return VaR, CVaR

# Cálculo del Drawdown máximo
def calcular_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod() - 1
    drawdown = cumulative_returns - cumulative_returns.cummax()
    max_drawdown = drawdown.min()  # Drawdown máximo
    return max_drawdown

# Cálculo del Sharpe Ratio
def calcular_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252  # Ajustamos por el riesgo libre
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

# Cálculo del Sortino Ratio
def calcular_sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < target_return]
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    return np.sqrt(252) * excess_returns.mean() / downside_deviation if downside_deviation != 0 else np.nan

# Función principal
def analizar_etfs(etfs, start_date, end_date):
    data = obtener_datos_acciones(etfs, start_date, end_date)  # Descargar los precios
    resultados = {}

    for etf in etfs:
        returns, media, sesgo, curtosis = calcular_metricas(data[etf])
        var_95, cvar_95 = calcular_var_cvar(returns)
        drawdown = calcular_drawdown(returns)
        sharpe = calcular_sharpe_ratio(returns)
        sortino = calcular_sortino_ratio(returns)

        # Almacenamos los resultados
        resultados[etf] = {
            "Media": media,
            "Sesgo": sesgo,
            "Curtosis": curtosis,
            "VaR 95%": var_95,
            "CVaR 95%": cvar_95,
            "Drawdown": drawdown,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino
        }

    # Convertimos los resultados a un DataFrame
    df_resultados = pd.DataFrame(resultados).T  # Transponemos para que cada ETF sea una fila
    return df_resultados, data

# Crear la interfaz en Streamlit

# Título de la app
st.title("Análisis de ETFs")

# Selección de ETFs
etfs = ["LQD", "EMB", "VTI", "EEM", "GLD"]  # Símbolos de los ETFs
selected_etfs = st.multiselect("Selecciona los ETFs para analizar", etfs, default=etfs)

# Selección de fechas
start_date = st.date_input("Fecha de inicio", datetime(2010, 1, 1))
end_date = st.date_input("Fecha de fin", datetime(2023, 12, 31))

# Botón para ejecutar el análisis
if st.button("Analizar ETFs"):
    # Analizamos los ETFs seleccionados
    df_resultados, data = analizar_etfs(selected_etfs, start_date, end_date)

    # Mostrar la tabla de resultados de rentabilidad media ordenada
    st.write("Rentabilidad Promedio de los ETFs seleccionados:")
    df_rentabilidad_ordenada = df_resultados[["Media"]].sort_values(by="Media", ascending=False)  # Ordenamos por la media
    st.dataframe(df_rentabilidad_ordenada.style.format({
        "Media": "{:.2%}"
    }))

    # Mostrar el análisis estadístico en una tabla
    st.write("Métricas estadísticas de los ETFs:")
    st.dataframe(df_resultados.style.format({
        "Media": "{:.2%}",
        "Sesgo": "{:.2f}",
        "Curtosis": "{:.2f}",
        "VaR 95%": "{:.2%}",
        "CVaR 95%": "{:.2%}",
        "Drawdown": "{:.2%}",
        "Sharpe Ratio": "{:.2f}",
        "Sortino Ratio": "{:.2f}"
    }))

    # Calcular los rendimientos diarios de los ETFs seleccionados
    st.write("Rendimientos diarios de los ETFs seleccionados:")
    df_rendimientos = data.pct_change().dropna()  # Calculamos los rendimientos diarios

    # Mostrar los rendimientos diarios en una tabla
    st.dataframe(df_rendimientos.style.format("{:.2%}"))

    # Mostrar gráfico de los precios históricos
    st.write("Gráfico de precios históricos de los ETFs seleccionados:")
    fig = go.Figure()

    for etf in selected_etfs:
        fig.add_trace(go.Scatter(x=data.index, y=data[etf], mode='lines', name=etf))

    fig.update_layout(title='Precios Históricos de los ETFs',
                      xaxis_title='Fecha',
                      yaxis_title='Precio de Cierre',
                      template='plotly_dark')
    st.plotly_chart(fig)
## INCISO B ##
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import scipy.optimize as sco
import streamlit as st
from datetime import date

# Símbolos de los ETFs
symbols = ['LQD', 'EMB', 'VTI', 'EEM', 'GLD']

# Número de activos
numofasset = len(symbols)

# Número de portafolios para simulación
numofportfolio = 10000

# Función para descargar datos históricos
def download_data(tickers, start_date='2010-01-01', end_date=date.today().strftime('%Y-%m-%d')):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Close']

# Descargar los datos de los ETFs
df = download_data(symbols, start_date='2010-01-01', end_date='2020-12-31')

# Calcular rendimientos diarios
returns = df.pct_change().fillna(0)

# Calcular retorno y volatilidad anualizados
annualized_return = round(returns.mean() * 252 * 100, 2)
annualized_volatility = round(returns.std() * np.sqrt(252) * 100, 2)

# Crear un DataFrame con estos valores
data = pd.DataFrame({
    'Annualized Return': annualized_return,
    'Annualized Volatility': annualized_volatility
})

# Función para simular portafolios
def portfolio_simulation(returns):
    rets = []
    vols = []
    wts = []

    for i in range(numofportfolio):
        weights = np.random.random(numofasset)[:, np.newaxis]
        weights /= np.sum(weights)
        rets.append(weights.T @ np.array(returns.mean() * 252)[:, np.newaxis])
        vols.append(np.sqrt(np.dot(np.dot(weights.T, returns.cov() * 252), weights)))
        wts.append(weights.flatten())

    portdf = 100 * pd.DataFrame({
        'port_rets': np.array(rets).flatten(),
        'port_vols': np.array(vols).flatten(),
        'weights': list(np.array(wts))
    })

    portdf['sharpe_ratio'] = portdf['port_rets'] / portdf['port_vols']
    return round(portdf, 2)

# Simular los portafolios
temp = portfolio_simulation(returns)

# Portafolio con el máximo Sharpe ratio
max_sharpe_portfolio = temp.iloc[temp.sharpe_ratio.idxmax()]
msrpwts = temp['weights'][temp['sharpe_ratio'].idxmax()]
max_sharpe_wts_dict = dict(zip(symbols, np.around(msrpwts, 2)))

# Portafolio con mínima volatilidad
min_volatility_portfolio = temp.iloc[temp.port_vols.idxmin()]
min_volatility_wts = temp['weights'][temp.port_vols.idxmin()]
min_volatility_wts_dict = dict(zip(symbols, np.around(min_volatility_wts, 2)))

# Función para calcular estadísticas del portafolio
def portfolio_stats(weights):
    weights = np.array(weights)[:, np.newaxis]
    port_rets = weights.T @ np.array(returns.mean() * 252)[:, np.newaxis]
    port_vols = np.sqrt(np.dot(np.dot(weights.T, returns.cov() * 252), weights))
    return np.array([port_rets, port_vols, port_rets / port_vols]).flatten()

# Optimización para el máximo Sharpe Ratio
def min_sharpe_ratio(weights):
    return -portfolio_stats(weights)[2]

bnds = tuple((0, 1) for x in range(numofasset))
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
initial_wts = numofasset * [1. / numofasset]

opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)
max_sharpe_wts = opt_sharpe['x']
max_sharpe_wts_dict_optimized = dict(zip(symbols, np.around(max_sharpe_wts, 2)))

# Optimización para mínima volatilidad
def min_variance(weights):
    return portfolio_stats(weights)[1]**2

opt_var = sco.minimize(min_variance, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)
min_volatility_wts = opt_var['x']
min_volatility_wts_dict_optimized = dict(zip(symbols, np.around(min_volatility_wts, 2)))

# Graficar la Frontera Eficiente
targetrets = np.linspace(0.02, 0.30, 100)
tvols = []

for tr in targetrets:
    ef_cons = ({
        'type': 'eq', 'fun': lambda x: portfolio_stats(x)[0] - tr
    }, {
        'type': 'eq', 'fun': lambda x: np.sum(x) - 1
    })
    opt_ef = sco.minimize(min_variance, initial_wts, method='SLSQP', bounds=bnds, constraints=ef_cons)
    tvols.append(np.sqrt(opt_ef['fun']))

targetvols = np.array(tvols)

efport = pd.DataFrame({
    'targetrets': np.around(100 * targetrets, 2),
    'targetvols': np.around(100 * targetvols, 2),
    'targetsharpe': np.around(targetrets / targetvols, 2)
})

# Visualización en Streamlit
st.title("Portafolio de Inversión: Optimización y Frontera Eficiente")

st.subheader("Portafolio con Máximo Sharpe Ratio")
st.write(max_sharpe_wts_dict_optimized)

st.subheader("Portafolio con Mínima Volatilidad")
st.write(min_volatility_wts_dict_optimized)

# Graficar la frontera eficiente
fig = px.scatter(
    efport, x='targetvols', y='targetrets', color='targetsharpe',
    labels={'targetrets': 'Expected Return (%)', 'targetvols': 'Expected Volatility (%)', 'targetsharpe': 'Sharpe Ratio'},
    title="Efficient Frontier Portfolio",
    color_continuous_scale=px.colors.sequential.Viridis
)

fig.update_layout(
    xaxis=dict(range=[efport['targetvols'].min() - 1, efport['targetvols'].max() + 1]),
    yaxis=dict(range=[efport['targetrets'].min() - 1, efport['targetrets'].max() + 1]),
    coloraxis_colorbar=dict(title='Sharpe Ratio')
)

fig.add_scatter(
    mode='markers',
    x=[100 * portfolio_stats(opt_sharpe['x'])[1]],
    y=[100 * portfolio_stats(opt_sharpe['x'])[0]],
    marker=dict(color='red', size=12, symbol='star'),
    name='Max Sharpe'
)

fig.add_scatter(
    mode='markers',
    x=[100 * portfolio_stats(opt_var['x'])[1]],
    y=[100 * portfolio_stats(opt_var['x'])[0]],
    marker=dict(color='green', size=12, symbol='star'),
    name='Min Variance'
)

st.plotly_chart(fig)
 ## c) ##
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
fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[symbols].dot(max_sharpe_wts),
                         mode='lines', name='Max Sharpe Portfolio'))

# Portafolio con mínima volatilidad
fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[symbols].dot(min_volatility_wts),
                         mode='lines', name='Min Volatility Portfolio'))

# Portafolio igualitario
equal_weights = np.ones(numofasset) / numofasset
fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[symbols].dot(equal_weights),
                         mode='lines', name='Equal Weight Portfolio'))

# S&P 500
fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns['^GSPC'],
                         mode='lines', name='S&P 500'))

fig.update_layout(title='Rendimiento Acumulado 2021-2023',
                  xaxis_title='Fecha',
                  yaxis_title='Rendimiento Acumulado')

# Mostrar la gráfica en Streamlit
st.plotly_chart(fig)

