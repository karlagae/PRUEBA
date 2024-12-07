import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

# Configuración de la aplicación
st.title("Optimización de Portafolios y Análisis de ETFs")
st.sidebar.header("Parámetros de Análisis")

# Selección de ETFs y fechas
etfs = st.sidebar.text_input("ETFs separados por comas", "SPY, QQQ, IWM, EFA, EEM")
start_date = st.sidebar.date_input("Fecha de inicio", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("Fecha de fin", pd.to_datetime("2023-12-31"))

# Descarga de datos
@st.cache_data
def download_data(symbols, start, end):
    data = yf.download(symbols, start=start, end=end)["Adj Close"]
    return data

symbols = [s.strip().upper() for s in etfs.split(",")]
try:
    prices = download_data(symbols, start_date, end_date)
    st.write(f"Datos cargados para: {', '.join(symbols)}")
except Exception as e:
    st.error(f"Error descargando datos: {e}")
    st.stop()

# Rendimientos diarios
returns = prices.pct_change().dropna()

# Cálculo de métricas para ETFs individuales
metrics = pd.DataFrame(index=symbols)
metrics["Mean Return"] = returns.mean() * 252
metrics["Volatility"] = returns.std() * np.sqrt(252)
metrics["Sharpe Ratio"] = metrics["Mean Return"] / metrics["Volatility"]

# Visualización de métricas
st.subheader("Métricas por ETF")
st.dataframe(metrics.style.format("{:.4f}"))

# Gráfico de precios históricos
st.subheader("Gráfico de precios históricos")
fig_prices = go.Figure()
for symbol in prices.columns:
    fig_prices.add_trace(go.Scatter(x=prices.index, y=prices[symbol], mode="lines", name=symbol))
fig_prices.update_layout(title="Precios Históricos", xaxis_title="Fecha", yaxis_title="Precio Ajustado")
st.plotly_chart(fig_prices)

# Simulación de portafolios
def simulate_portfolios(returns, num_portfolios=10000):
    num_assets = returns.shape[1]
    results = np.zeros((4, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        portfolio_return = np.sum(weights * returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio
        results[3, i] = i  # Index for weights tracking
    
    results_df = pd.DataFrame(results.T, columns=["Return", "Volatility", "Sharpe Ratio", "Index"])
    return results_df, weights_record

portfolios, weights = simulate_portfolios(returns)
max_sharpe = portfolios.iloc[portfolios["Sharpe Ratio"].idxmax()]
min_volatility = portfolios.iloc[portfolios["Volatility"].idxmin()]

# Frontera eficiente
st.subheader("Frontera Eficiente")
fig_efficient = go.Figure()
fig_efficient.add_trace(go.Scatter(x=portfolios["Volatility"], y=portfolios["Return"], 
                                    mode="markers", marker=dict(color=portfolios["Sharpe Ratio"], colorscale="Viridis", size=5), name="Portafolios Simulados"))
fig_efficient.add_trace(go.Scatter(x=[max_sharpe["Volatility"]], y=[max_sharpe["Return"]], 
                                    mode="markers", marker=dict(color="red", size=10), name="Máximo Sharpe"))
fig_efficient.add_trace(go.Scatter(x=[min_volatility["Volatility"]], y=[min_volatility["Return"]], 
                                    mode="markers", marker=dict(color="blue", size=10), name="Mínima Volatilidad"))
fig_efficient.update_layout(title="Frontera Eficiente", xaxis_title="Volatilidad", yaxis_title="Retorno")
st.plotly_chart(fig_efficient)

# Comparación de métricas
st.subheader("Comparación de Portafolios")
comparison = pd.DataFrame({
    "Portafolio": ["Máximo Sharpe", "Mínima Volatilidad"],
    "Retorno Anualizado": [max_sharpe["Return"], min_volatility["Return"]],
    "Volatilidad Anualizada": [max_sharpe["Volatility"], min_volatility["Volatility"]],
    "Sharpe Ratio": [max_sharpe["Sharpe Ratio"], min_volatility["Sharpe Ratio"]],
})
st.dataframe(comparison.style.format("{:.4f}"))

# Gráfico de rendimientos acumulados
cumulative_returns = (1 + returns).cumprod()
st.subheader("Rendimientos Acumulados")
fig_cum_returns = go.Figure()
for symbol in cumulative_returns.columns:
    fig_cum_returns.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[symbol], mode="lines", name=symbol))
fig_cum_returns.update_layout(title="Rendimientos Acumulados", xaxis_title="Fecha", yaxis_title="Rendimiento Acumulado")
st.plotly_chart(fig_cum_returns)
