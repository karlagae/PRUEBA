# Importar bibliotecas necesarias
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return

# Función para realizar la optimización con Black-Litterman
def optimizar_black_litterman(etfs, start_date, end_date, tau=0.05, Q=None, P=None):
    precios = obtener_datos_acciones(etfs, start_date, end_date)
    retornos = precios.pct_change().dropna()

    # Calcular los retornos esperados y la matriz de covarianza
    mu = mean_historical_return(precios)
    S = CovarianceShrinkage(precios).ledoit_wolf()

    # Crear frontera eficiente y optimizar
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    pesos = ef.clean_weights()
    retornos_post = ef.portfolio_performance(verbose=True)

    return pesos, retornos_post

# Nueva pestaña para Black-Litterman
tab3, = st.tabs(["Optimización Black-Litterman"])

with tab3:
    st.header("Optimización de Portafolio con Black-Litterman")
    
    # Entrada de datos del usuario
    etfs_input = st.text_input("Ingrese los símbolos de los activos (por ejemplo: LQD,EMB,VTI,EEM,GLD):", "LQD,EMB,VTI,EEM,GLD")
    etfs = [etf.strip() for etf in etfs_input.split(',')]
    tau = st.slider("Seleccione el valor de tau (incertidumbre del mercado):", 0.01, 0.2, 0.05)
    
    Q_input = st.text_area("Ingrese la matriz Q (en formato de lista, ejemplo: [[0.02], [-0.01]]):", "[[0.02], [-0.01]]")
    P_input = st.text_area("Ingrese la matriz P (en formato de lista, ejemplo: [[1, 0, -1, 0, 0], [0, 1, 0, -1, 0]]):", "[[1, 0, -1, 0, 0], [0, 1, 0, -1, 0]]")
    
    Q = np.array(eval(Q_input))
    P = np.array(eval(P_input))
    
    # Selección de fechas
    start_date = st.date_input("Fecha de inicio:", value=pd.Timestamp("2010-01-01"))
    end_date = st.date_input("Fecha de fin:", value=pd.Timestamp("2023-12-31"))
    
    # Optimizar y mostrar resultados
    if st.button("Optimizar"):
        with st.spinner("Optimizando portafolio..."):
            pesos_optimos, retornos_post = optimizar_black_litterman(etfs, start_date, end_date, tau, Q, P)
        
        # Mostrar resultados
        st.subheader("Pesos Óptimos del Portafolio")
        st.write(pesos_optimos)
        
        st.subheader("Desempeño del Portafolio")
        st.write(f"Rendimiento esperado: {retornos_post[0]:.2%}")
        st.write(f"Volatilidad: {retornos_post[1]:.2%}")
        st.write(f"Ratio de Sharpe: {retornos_post[2]:.2f}")
        
        # Gráfico de barras para pesos
        fig_pesos = go.Figure()
        fig_pesos.add_trace(go.Bar(x=list(pesos_optimos.keys()), y=list(pesos_optimos.values())))
        fig_pesos.update_layout(title="Composición del Portafolio Óptimo", xaxis_title="Activos", yaxis_title="Peso")
        st.plotly_chart(fig_pesos, use_container_width=True)
        
        # Comparación con benchmark
        precios = obtener_datos_acciones(etfs, start_date, end_date)
        retornos = precios.pct_change().dropna()
        pesos_lista = list(pesos_optimos.values())
        retornos_acumulados_portafolio = (1 + np.dot(retornos, pesos_lista)).cumprod()
        
        indice = yf.download('^GSPC', start=start_date, end=end_date)['Close']
        retornos_indice = indice.pct_change().dropna()
        retornos_acumulados_indice = (1 + retornos_indice).cumprod()
        
        # Gráfico de comparación de retornos acumulados
        fig_comparacion = go.Figure()
        fig_comparacion.add_trace(go.Scatter(x=precios.index, y=retornos_acumulados_portafolio, name='Portafolio Óptimo'))
        fig_comparacion.add_trace(go.Scatter(x=indice.index, y=retornos_acumulados_indice, name='S&P 500'))
        fig_comparacion.update_layout(title="Retornos Acumulados", xaxis_title="Fecha", yaxis_title="Retorno Acumulado")
        st.plotly_chart(fig_comparacion, use_container_width=True)
