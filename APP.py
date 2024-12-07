import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu
from pypfopt import BlackLittermanModel, risk_models, expected_returns, EfficientFrontier

# Configuración de la página
st.set_page_config(page_title="Análisis de ETFs", layout="wide")

# Menú de navegación
with st.sidebar:
    menu = option_menu("Menú", ["Inicio", "Gráficas Interactivas", "Black-Litterman"], 
                       icons=["house", "chart-bar", "calculator"], menu_icon="cast", default_index=0)

# Función para obtener datos simulados o cargados
@st.cache
def cargar_datos(etfs, fecha_inicio, fecha_fin):
    # Aquí puedes reemplazar esto con la carga de datos real
    fechas = pd.date_range(fecha_inicio, fecha_fin, freq='B')
    datos = {etf: np.cumsum(np.random.randn(len(fechas))) for etf in etfs}
    return pd.DataFrame(datos, index=fechas)

# Página de inicio
if menu == "Inicio":
    st.title("Análisis de ETFs")
    st.write("Selecciona los ETFs para analizar.")
    
    etfs = st.multiselect("Selecciona los ETFs", ["LQD", "EMB", "VTI", "EEM", "GLD"], default=["LQD", "EMB"])
    fecha_inicio = st.date_input("Fecha de inicio", value=pd.to_datetime("2010-01-01"))
    fecha_fin = st.date_input("Fecha de fin", value=pd.to_datetime("2023-12-31"))
    
    if st.button("Analizar ETFs"):
        datos = cargar_datos(etfs, fecha_inicio, fecha_fin)
        st.line_chart(datos)

# Gráficas Interactivas
elif menu == "Gráficas Interactivas":
    st.title("Gráficas Interactivas")
    etfs = ["LQD", "EMB", "VTI", "EEM", "GLD"]
    datos = cargar_datos(etfs, "2010-01-01", "2023-12-31")
    
    st.subheader("Precios históricos")
    fig = px.line(datos, x=datos.index, y=datos.columns, title="Evolución de precios")
    st.plotly_chart(fig, use_container_width=True)

# Black-Litterman
elif menu == "Black-Litterman":
    st.title("Modelo Black-Litterman")
    
    # Datos simulados
    etfs = ["LQD", "EMB", "VTI", "EEM", "GLD"]
    datos = cargar_datos(etfs, "2010-01-01", "2023-12-31")
    mu = expected_returns.mean_historical_return(datos)
    S = risk_models.sample_cov(datos)
    
    # Input del usuario para creencias
    st.subheader("Define tus creencias")
    view_1 = st.text_input("Vista 1 (e.g., LQD > EMB)", "LQD > EMB")
    view_2 = st.text_input("Vista 2 (e.g., VTI > GLD)", "VTI > GLD")
    confidencias = st.slider("Nivel de confianza en tus vistas (0-1)", 0.0, 1.0, 0.5)
    
    # Configuración de Black-Litterman
    bl = BlackLittermanModel(S, absolute_views={etfs[0]: confidencias, etfs[1]: -confidencias})
    bl_return = bl.bl_returns()
    ef = EfficientFrontier(bl_return, S)
    pesos = ef.max_sharpe()
    
    st.subheader("Pesos Óptimos")
    st.write(pesos)
    
    st.subheader("Gráfica de Asignaciones")
    pesos_df = pd.DataFrame(list(pesos.items()), columns=["Activo", "Peso"])
    fig = px.bar(pesos_df, x="Activo", y="Peso", title="Asignación de Portafolio")
    st.plotly_chart(fig, use_container_width=True)


