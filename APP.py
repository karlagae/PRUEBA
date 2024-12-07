# ---------------------------------------------------------------------------------------------------#
#                                      OPTIMIZATION PAGE

import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotly import graph_objs as go

# Inicializar variables en session_state si no existen
st.session_state.setdefault("returns", None)
st.session_state.setdefault("resultados_pesos", None)
st.session_state.setdefault("metricas", None)

# Verificar que los datos necesarios están disponibles
if st.session_state.returns is not None and st.session_state.resultados_pesos is not None:
    st.markdown("## Optimization Results")
    st.text(
        "The table below contains the weights of each portfolio optimized using "
        "three methods: Minimum Volatility, Maximum Sharpe Ratio, and Target Volatility. "
        "Analyze the weights for the selected optimization methods."
    )

    # Convertir pesos de resultados a un DataFrame para mostrarlo
    weights_df = pd.DataFrame(
        st.session_state.resultados_pesos, index=st.session_state.returns.columns
    )

    # Agregar el nombre de la estrategia como columna
    weights_df.reset_index(inplace=True)
    weights_df.rename(columns={"index": "Asset"}, inplace=True)

    # Redondear valores a 2 decimales
    weights_df.iloc[:, 1:] = weights_df.iloc[:, 1:].round(2)

    # Mostrar tabla de pesos en Streamlit
    st.dataframe(weights_df.style.format("{:.2%}"), use_container_width=True)

    # Graficar los pesos de los portafolios optimizados
    st.subheader("Portfolio Weights")
    melted_weights = pd.melt(
        weights_df, id_vars=["Asset"], var_name="Portfolio", value_name="Weight"
    )

    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("mako", n_colors=len(st.session_state.resultados_pesos))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=melted_weights, x="Portfolio", y="Weight", hue="Asset", palette=colors, ax=ax
    )
    ax.set_title("Weights per Portfolio", fontsize=16)
    ax.set_ylabel("Weight (%)", fontsize=12)
    ax.set_xlabel("Portfolio", fontsize=12)
    ax.legend(loc="upper right", title="Assets")
    st.pyplot(fig)
    plt.close()

    # Ver estadísticas del portafolio
    st.subheader("Portfolio Statistics")
    st.text(
        "Compare the performance of each optimized portfolio using key metrics such as "
        "expected return, volatility, and Sharpe ratio."
    )

    # Calcular estadísticas del portafolio
    portfolio_stats = st.session_state.metricas(
        st.session_state.resultados_pesos, st.session_state.returns
    )
    st.dataframe(portfolio_stats.style.format("{:.2f}"), use_container_width=True)

    # Botón para proceder al backtesting
    st.text("Ready? Proceed to backtesting!")
    col1, col2 = st.columns([1, 0.2])
    with col2:
        if st.button("Backtesting!"):
            st.switch_page("Backtesting")
else:
    st.warning("No data available. Please return to the previous step and load the required data.")

# ---------------------------------------------------------------------------------------------------#
#                                      BACKTESTING

# Inicializar variables en session_state si no existen
st.session_state.setdefault("returns_sp500", None)
st.session_state.setdefault("returns_bt", None)

if st.session_state.resultados_pesos is not None:
    st.markdown("## Backtesting :mag:")
    st.text(
        "In this section, you can backtest the optimized portfolios within your selected date range. "
        "Ensure your backtest dates fall within the available data. Full-year backtests are recommended "
        "for more accurate results."
    )

    # Selección de fechas para el backtesting
    start_bt = st.date_input(
        "Backtesting start:",
        value=dt.date(2021, 1, 1),
        min_value=st.session_state.start_date,
        max_value=st.session_state.end_date,
    )
    end_bt = st.date_input(
        "Backtesting end:",
        value=dt.date(2024, 1, 1),
        min_value=st.session_state.start_date,
        max_value=st.session_state.end_date,
    )

    # Filtrar retornos para el rango de fechas seleccionado
    if st.session_state.returns is not None:
        returns_bt = st.session_state.returns.loc[start_bt:end_bt]
        st.session_state.returns_bt = returns_bt

    if st.button("Backtest!"):
        # Descargar datos del S&P 500
        sp500 = (
            st.session_state.get_asset_data("^GSPC", start_bt, end_bt)
            .rename(columns={"Close": "^GSPC"})
        )
        # Convertir divisas según la moneda objetivo
        sp500 = st.session_state.convert_to_currency(
            sp500, start_bt, end_bt, st.session_state.target_currency
        )
        # Calcular retornos diarios
        sp500["^GSPC"] = sp500["^GSPC"].pct_change().dropna()
        st.session_state.returns_sp500 = sp500[["^GSPC"]]

# Manejo de posibles errores de inicialización
if st.session_state.returns_bt is not None and st.session_state.returns_sp500 is not None:
    st.subheader("Annual Returns")
    st.write("Aquí iría el análisis de retornos.")
























    
    # Valor acumulado diario
    portfolio_value = (1 + results_daily / 100).cumprod() * 100
    fig_daily = go.Figure()
    for col in portfolio_value.columns:
        fig_daily.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value[col], mode="lines", name=col))
    fig_daily.update_layout(title="Portfolio Value", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig_daily, use_container_width=True)

    # Estadísticas de backtesting
    st.subheader("Backtesting Statistics")
    rf_rate = st.session_state.rf_rate_us.iloc[-1] if st.session_state.rf_use == "United States" else st.session_state.rf_rate_mx.iloc[-1]
    stats = st.session_state.metricas(results_daily, rf_rate=rf_rate)
    st.dataframe(stats)

    # Botón para continuar
    if st.button("Black-Litterman!"):
        st.switch_page("pages/3_Black-Litterman Model.py")

