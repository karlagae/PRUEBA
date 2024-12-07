import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
import scipy.optimize as sco

# Streamlit Sidebar
st.sidebar.title("Análisis de ETFs")
st.sidebar.markdown("Selecciona los parámetros de tu análisis.")

# Sidebar input for ETFs selection
etfs = ["LQD", "EMB", "VTI", "EEM", "GLD"]
selected_etfs = st.sidebar.multiselect("Selecciona los ETFs para analizar", etfs, default=etfs)

# Sidebar input for date range selection using sliders
start_date = st.sidebar.date_input("Fecha de inicio", datetime(2010, 1, 1))
end_date = st.sidebar.date_input("Fecha de fin", datetime.today())

# Button to run the analysis
run_analysis = st.sidebar.button("Ejecutar Análisis")

# Function to fetch data
def obtener_datos_acciones(simbolos, start_date, end_date):
    data = yf.download(simbolos, start=start_date, end=end_date)['Close']
    return data.ffill().dropna()

# Function to calculate statistics
def calcular_metricas(df):
    returns = df.pct_change().dropna()
    media = returns.mean()
    sesgo = returns.skew()
    curtosis = returns.kurtosis()
    return returns, media, sesgo, curtosis

# Main analysis
if run_analysis:
    data = obtener_datos_acciones(selected_etfs, start_date, end_date)

    # Calculate metrics for each ETF
    resultados = {}
    for etf in selected_etfs:
        returns, media, sesgo, curtosis = calcular_metricas(data[etf])
        resultados[etf] = {
            "Media": media,
            "Sesgo": sesgo,
            "Curtosis": curtosis
        }

    df_resultados = pd.DataFrame(resultados).T
    st.write("Métricas de los ETFs seleccionados:")
    st.dataframe(df_resultados.style.format({
        "Media": "{:.2%}",
        "Sesgo": "{:.2f}",
        "Curtosis": "{:.2f}"
    }))

    # Plotting the price data
    st.write("Precios históricos de los ETFs seleccionados:")
    fig = go.Figure()

    for etf in selected_etfs:
        fig.add_trace(go.Scatter(x=data.index, y=data[etf], mode='lines', name=etf))

    fig.update_layout(title='Precios Históricos de los ETFs',
                      xaxis_title='Fecha',
                      yaxis_title='Precio de Cierre',
                      template='plotly_dark')
    st.plotly_chart(fig)

    # Portfolio optimization simulation (Sharpe ratio)
    returns = data.pct_change().dropna()
    numofasset = len(selected_etfs)
    numofportfolio = 10000

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

    # Simulate portfolios
    temp = portfolio_simulation(returns)

    # Display max Sharpe ratio portfolio
    max_sharpe_portfolio = temp.iloc[temp.sharpe_ratio.idxmax()]
    msrpwts = temp['weights'][temp['sharpe_ratio'].idxmax()]
    st.write("Portafolio con el Máximo Sharpe Ratio:", dict(zip(selected_etfs, np.around(msrpwts, 2))))

    # Plot Efficient Frontier
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

    fig = go.Figure(data=go.Scatter(x=efport['targetvols'], y=efport['targetrets'], mode='lines', name='Frontera Eficiente'))
    st.plotly_chart(fig)

