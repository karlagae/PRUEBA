import pandas as pd
import numpy as np
#from numpy import *
from numpy.linalg import multi_dot
import scipy as stats
from scipy.stats import kurtosis, skew, norm
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go

#---------------------------------------------------------------------------------------------------#
#                                             PAGE INFO

# esta pagina es la segunda pagina de la aplicacion
# aqui se optimizan los portafolios usando minima volatilidad, maximo sr, y minima volatilidad obj
# tambien se realiza el backtesting de los portafolios optimizados

st.set_page_config(
    page_title="Portfolio Optimization",
    page_icon = "mag"
)
st.title("Portfolio Optimization & Backtesting")

#---------------------------------------------------------------------------------------------------#

# verificamos que tengamos disponible la informacion necesaria para optimizar los portafolios

if ("data" in st.session_state and st.session_state.data is not None) and \
    ("returns" in st.session_state and st.session_state.returns is not None):
    st.success("Session data successfully loaded.")
    st.write("Closing prices:")
    st.dataframe(st.session_state.data.tail())
    st.write("Daily returns:")
    st.dataframe(st.session_state.returns.tail())
else:
    st.warning("No data available. Please return to the main page to load the required data.")
    st.stop()

#---------------------------------------------------------------------------------------------------#
#                                 OPTIMIZACION DE PORTAFOLIOS

import scipy.optimize as sco

# Definimos la funcion portfolio stats para calular retornos, volatilidad y 
# sharpe ratiode los portafolios
def portfolio_stats(weights, returns, return_df = False):
    weights = np.array(weights)[:,np.newaxis]
    port_rets = weights.T @ np.array(returns.mean() * 252)[:,np.newaxis]
    port_vols = np.sqrt(multi_dot([weights.T, returns.cov() * 252, weights]))
    sharpe_ratio = port_rets/port_vols
    resultados = np.array([port_rets, port_vols, sharpe_ratio]).flatten()
    
    if return_df == True:
        return pd.DataFrame(data = np.round(resultados,4),
                            index = ["Returns", "Volatility", "Sharpe_Ratio"],
                            columns = ["Resultado"])
    else:
        return resultados

st.markdown("## Optimization :muscle:")

# Definimos las fechas sobre las que queremos optimizar el portafolio
st.text("Define the date range for portfolio optimization before proceeding. (You can use the arrows in your keyboard \
        to change the dates.)")
opt_range = st.slider("Select a date range:", min_value = st.session_state.start_date, 
                      max_value = st.session_state.end_date, value=(st.session_state.start_date, st.session_state.end_date),
                      format="YYYY-MM-DD") 
# acceder a las fehcas seleccionadas
st.session_state.start_date_opt, st.session_state.end_date_opt = opt_range

st.write("Start:", st.session_state.start_date_opt)
st.write("End:", st.session_state.end_date_opt)

if "returns1" not in st.session_state:
    st.session_state.returns1 = None

# Guardamos los retornos en un nuevo df
if st.session_state.returns is not None:
    st.session_state.returns1 = st.session_state.returns.loc[st.session_state.start_date_opt:st.session_state.end_date_opt]

# ingresar el rendimiento objetivo del portafolio de minima varianza con rendimiento objetivo
r_obj = st.number_input(
    "Specify the target return for the minimum volatility portfolio:",
    value = 0.1, min_value = 0.0, max_value= 1.0
    )

opt_bool = False
if st.button("Go!"):
    opt_bool = True

## 1. MINIMA VOLATILIDAD ---------------------------------------------------------------------------

# definimos la funcion que nos ayudara a obtener la volatilidad del portafolio con portfolio_stats
def get_volatility(weights, returns):
    return portfolio_stats(weights, returns)[1]

# vamos a definir una funcion que optimice el portafolio bajo minima volatilidad
def min_vol_opt(returns):
    # Definimos las condiciones para nuestra optimizacion
        # la suma de los activos debe ser 1
        # el peso de cada activo debe estar entre 0 y 1
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    bnds = tuple((0, 1) for x in range(len(returns.columns)))
    
    # usaremos un portafolio equaly weighted como pesos iniciales
    initial_wts = np.array(len(returns.columns)*[1./len(returns.columns)])
    
    # Usamos la funcion minimizar de scipy
    opt_vol = sco.minimize(fun=get_volatility, x0=initial_wts, args=(returns),
                           method='SLSQP', bounds=bnds, constraints=cons)
    
    # obtenemos los pesos del portafolio bajo la optimizacion
    #min_vol_pesos = list(zip(returns.columns, np.around(opt_vol['x']*100,2)))
    min_vol_pesos = pd.DataFrame(data = np.around(opt_vol['x']*100,2),
                                 index = returns.columns,
                                 columns = ["Min_Vol"]) 
    
    # obtenemos las estadisticas del portafolio optimizado
    min_vol_stats = portfolio_stats(opt_vol['x'], returns, return_df = True)
    min_vol_stats = min_vol_stats.rename(columns={"Resultado":"Min_Vol"})
    
    return {"min_vol_pesos": min_vol_pesos, "min_vol_stats": min_vol_stats}

if "min_vol_resultados" not in st.session_state:
    st.session_state.min_vol_resultados = None

if opt_bool:
    if st.session_state.returns1 is not None:
        try:         
            st.session_state.min_vol_resultados = min_vol_opt(st.session_state.returns1)
            st.success("Minimum Volatility Portfolio successfully optimized!")   
        except:
            st.warning("An error occurred while optimizing the Minimum Volatility Portfolio.")

if st.session_state.min_vol_resultados is not None:
    st.subheader("Minimum Volatility Portfolio")
    st.markdown("Weights: :weight_lifter:") 
    st.dataframe(st.session_state.min_vol_resultados["min_vol_pesos"])
    st.markdown("Stats: :money_with_wings:") 
    st.dataframe(st.session_state.min_vol_resultados["min_vol_stats"])

## 2. MAX Sharpe Ratio -----------------------------------------------------------------------------

# ya que en esta optimizacion buscamos maximizar el sharpe ratio, definiremos una funcion
# de apoyo que obtendra el valor negativo del sharpe ratio calculado en la funcion
# portfolio stats para poder usar la funcion de miniminzacion de scipy
def min_sharpe_ratio(weights, returns):
    return -portfolio_stats(weights, returns)[2]

#type(min_sharpe_ratio(min_vol_resultados["min_vol_pesos"]["Min_Vol"]/100, returns1))
#min_sharpe_ratio(min_vol_resultados["min_vol_pesos"]["Min_Vol"]/100, returns1)

# definimos la funcion que hara la optimizacion
def max_sr_opt(returns):
    # Definimos las condiciones para nuestra optimizacion
        # la suma de los activos debe ser 1
        # el peso de cada activo debe estar entre 0 y 1
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    bnds = tuple((0, 1) for x in range(len(returns.columns)))
    
    # usaremos un portafolio equaly weighted como pesos iniciales
    initial_wts = np.array(len(returns.columns)*[1./len(returns.columns)])
    
    # Usamos la funcion minimizar de scipy
    opt_sr = sco.minimize(fun=min_sharpe_ratio, x0=initial_wts, args=(returns),
                           method='SLSQP', bounds=bnds, constraints=cons)
    
    # obtenemos los pesos del portafolio bajo la optimizacion
    #min_vol_pesos = list(zip(returns.columns, np.around(opt_vol['x']*100,2)))
    max_sr_pesos = pd.DataFrame(data = np.around(opt_sr['x']*100,2),
                                 index = returns.columns,
                                 columns = ["Max_SR"]) 
    
    # obtenemos las estadisticas del portafolio optimizado
    max_sr_stats = portfolio_stats(opt_sr['x'], returns, return_df = True)
    max_sr_stats = max_sr_stats.rename(columns={"Resultado":"Max_SR"})
    
    return {"max_sr_pesos": max_sr_pesos, "max_sr_stats": max_sr_stats}


if "max_sr_resultados" not in st.session_state:
    st.session_state.max_sr_resultados = None

if opt_bool:
    if st.session_state.returns1 is not None:
        try:        
            st.session_state.max_sr_resultados = max_sr_opt(st.session_state.returns1)         
            st.success("Maximum Sharpe Ratio Portfolio successfully optimized!")           
        except:
            st.warning("An error occurred while optimizing the Maximum Sharpe Ratio Portfolio.")

if st.session_state.max_sr_resultados is not None:
    st.subheader("Maximum Sharpe Ratio Portfolio")
    st.markdown("Weights: :weight_lifter:") 
    st.dataframe(st.session_state.max_sr_resultados["max_sr_pesos"])
    st.markdown("Stats: :money_with_wings:") 
    st.dataframe(st.session_state.max_sr_resultados["max_sr_stats"])


## 3. Minima Volatilidad con Objetivo de Rendimiento -----------------------------------------------

# definimos la funcion para optimizar el portafolio
def min_vol_obj_opt(returns, r_obj):
    # definimos las condiciones para la optimizacion
    cons = ({'type': 'eq', 'fun': lambda x: portfolio_stats(x, returns)[0] - r_obj},
                   {'type': 'eq', 'fun': lambda x: sum(x) - 1})
    bnds = tuple((0, 1) for x in range(len(returns.columns)))
    
    # usaremos un portafolio equaly weighted como pesos iniciales
    initial_wts = np.array(len(returns.columns)*[1./len(returns.columns)])
    
    # Usamos la funcion minimizar de scipy
    opt_min_obj = sco.minimize(fun=get_volatility, x0=initial_wts, args=(returns),
                           method='SLSQP', bounds=bnds, constraints=cons)
    
    # obtenemos los pesos del portafolio bajo la optimizacion
    #min_vol_pesos = list(zip(returns.columns, np.around(opt_vol['x']*100,2)))
    min_obj_pesos = pd.DataFrame(data = np.around(opt_min_obj['x']*100,2),
                                 index = returns.columns,
                                 columns = ["Min_Vol_Obj"]) 
    
    # obtenemos las estadisticas del portafolio optimizado
    min_obj_stats = portfolio_stats(opt_min_obj['x'], returns, return_df = True)
    min_obj_stats = min_obj_stats.rename(columns={"Resultado":"Min_Vol_Obj"})
    
    return {"min_obj_pesos": min_obj_pesos, "min_obj_stats": min_obj_stats}

if "min_obj_resultados" not in st.session_state:
    st.session_state.min_obj_resultados = None

if opt_bool:
    if st.session_state.returns1 is not None:
        try:
            st.session_state.min_obj_resultados = min_vol_obj_opt(st.session_state.returns1, r_obj)
            st.success("Minimum Volatility Portfolio with Target Return successfully optimized!")    
        except:
            st.warning("An error occurred while optimizing the Minimum Volatility Portfolio with Target Return.")

if st.session_state.min_obj_resultados is not None:
    st.subheader("Minimum Volatility with Target Returns Portfolio")
    st.markdown("Weights: :weight_lifter:") 
    st.dataframe(st.session_state.min_obj_resultados["min_obj_pesos"])
    st.markdown("Stats: :money_with_wings:") 
    st.dataframe(st.session_state.min_obj_resultados["min_obj_stats"])

## Resultados --------------------------------------------------------------------------------------

if "resultados_pesos" not in st.session_state:
    st.session_state.resultados_pesos = None

if st.session_state.min_vol_resultados is not None and st.session_state.max_sr_resultados is not None and \
st.session_state.min_obj_resultados is not None:
    st.subheader("Outcome")
    chart_names = ["Minimum Volatility Weights", "Maximum Sharpe Ratio Weights", "Minimun Volatility with Target Weights"]
    try:
        # comparamos los pesos de cada activo en los portafolios
        resultados_pesos = st.session_state.min_vol_resultados["min_vol_pesos"].\
            merge(st.session_state.max_sr_resultados["max_sr_pesos"], on = "Ticker").\
            merge(st.session_state.min_obj_resultados["min_obj_pesos"], on = "Ticker")
        st.session_state.resultados_pesos = resultados_pesos
        #st.dataframe(st.session_state.resultados_pesos)

        # visualizamos los datos
        #i = 0
        color_map = dict(zip(st.session_state.resultados_pesos.index, st.session_state.colors))
        n_charts = len(st.session_state.resultados_pesos.columns)
        fig, axes = plt.subplots(nrows=1, ncols=n_charts, figsize=(5 * n_charts, 7))
        if n_charts == 1:
            axes = [axes]
        for i, (col, ax) in enumerate(zip(st.session_state.resultados_pesos.columns, axes)):

            # Filtrar valores positivos y sus correspondientes etiquetas
            filtered_values = st.session_state.resultados_pesos[col][st.session_state.resultados_pesos[col] > 0]
            filtered_labels = filtered_values.index  # Etiquetas para valores positivos

            # Obtener los colores correspondientes a los tickers filtrados
            temp_colors = [color_map[label] for label in filtered_labels]

            # Crear el gráfico de pie
            wedges, texts, autotexts = ax.pie(
                filtered_values, 
                labels=None,
                autopct='%1.1f%%', 
                pctdistance = 1.15,
                startangle=140,
                colors=temp_colors,
                wedgeprops={'edgecolor': 'white'}
            )

            # Agregar el título al subplot
            ax.set_title(f"{chart_names[i]}", fontsize=15)

            # Ajustar el tamaño de los valores porcentuales
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_color("black")
                #autotext.set_weight("bold")

            # Agregar leyenda al subplot
        fig.legend(
            wedges,  # Asignar los pedazos del gráfico a la leyenda
            filtered_labels,  # Etiquetas para la leyenda
            loc='lower center', 
            bbox_to_anchor=(0.5, 0.05), 
            fontsize=13,
            title="", 
            title_fontsize=10,
            frameon = False,
            ncol = len(st.session_state.resultados_pesos)
            )

        # Ajustar los márgenes para que no se solapen los gráficos
        plt.tight_layout()

        # Mostrar el gráfico completo
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"An error occurred while processing: {e}")

    try:
        # comparamos las metricas de cada portafolio
        resultados_stats = st.session_state.min_vol_resultados["min_vol_stats"].\
            join(st.session_state.max_sr_resultados["max_sr_stats"]).\
            join(st.session_state.min_obj_resultados["min_obj_stats"]).reset_index()
        
        st.session_state.resultados_stats = resultados_stats.rename(columns = {'index': 'Metrics',
                                                                               'Min_Vol': 'Minimum Volatility',
                                                                               'Max_SR': 'Maximum Sharpe Ratio',
                                                                               'Min_Vol_Obj': 'Min. Vol. Target'})
        stats_graf_df = pd.melt(st.session_state.resultados_stats, id_vars = "Metrics", var_name = "Portfolio",
                                value_name= "Values")
        # visualizamos los datos
        g = sns.catplot(data = stats_graf_df, x = 'Metrics', y = "Values", hue = 'Metrics', col = "Portfolio",
                    kind = "bar", height = 7, aspect = 0.7, palette=sns.color_palette("mako", n_colors=3), legend = False)
        g.set_titles("{col_name}")

        # Agregar etiquetas encima de las barras
        for ax in g.axes.flat:
            for container in ax.containers:  # Recorrer cada conjunto de barras
                ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=12, color="black")  # Etiquetas fuera de las barras

        g.fig.suptitle("Portfolio Metrics Comparison", fontsize=18, y=1.02)
        plt.subplots_adjust(top=0.9)
        g.set_axis_labels("", "", fontsize=19)
        st.pyplot(g)
        plt.close()

    except Exception as e:
        st.error(f"An error occurred while processing: {e}")



#---------------------------------------------------------------------------------------------------#
#                                      BACKTESTING

# agregar boton de backtesging

if st.session_state.resultados_pesos is not None:
    st.markdown("## Backtesting :mag:")
    st.text("In this section, you can backtest the optimized portfolios within your selected date range. \
            Ensure your backtest dates fall within the available data. Full-year backtests are recommended \
            for more accurate results.")

# Realizaremos backtesging de los portafolios optimizados en el punto anterior y compararemos
# los resultados con el S&P 500

if "returns_sp500" not in st.session_state:
    st.session_state.returns_sp500 = None
if "returns_bt" not in st.session_state:
    st.session_state.returns_bt = None

if st.session_state.resultados_pesos is not None:
    # Definimos las fechas sobre las que realizaremos el backtesting
    start_bt = st.date_input("Backtesting start:", value = dt.date(2021, 1, 1),
                              min_value = st.session_state.start_date, max_value=st.session_state.end_date)
    end_bt = st.date_input("Backtesting end:", value = dt.date(2024, 1, 1),
                           min_value = st.session_state.start_date, max_value=st.session_state.end_date)
    # seleccionamos los retornos entre las fechas seleccionadas
    returns_bt = st.session_state.returns.loc[start_bt:end_bt]
    st.session_state.returns_bt = returns_bt

    if st.button("Backtest!"):
        # descargamos los datos del S&P 500
        sp500 = st.session_state.get_asset_data("^GSPC", start_bt, end_bt).rename(columns = {'Close':'^GSPC'})
        # conversion de divisas
        sp500 = st.session_state.convert_to_currency(sp500, start_bt, end_bt, st.session_state.target_currency)
        # obtenemos los retornos diarios
        returns_sp500 = sp500.copy().sort_index()
        for columna in returns_sp500.columns:
            returns_sp500[columna] = (returns_sp500[columna] - returns_sp500[columna].shift(1)) / returns_sp500[columna].shift(1)

        returns_sp500 = returns_sp500.dropna()
        st.session_state.returns_sp500 = returns_sp500

if st.session_state.returns_bt is not None and st.session_state.returns_sp500 is not None:
    st.subheader("Annual Returns")  
    ## 1. Retornos Anuales

    # copiamos los retornos para obtener los datos anuales
    returns_bt_y = st.session_state.returns_bt.copy()
    # Asegurarse de que el índice sea de tipo datetime
    returns_bt_y.index = pd.to_datetime(returns_bt_y.index)
    # obtenemos los retornos anuales de cada activo
    returns_bt_y = (returns_bt_y.resample('YE').apply(lambda x: (1 + x).prod() - 1))
    #returns_bt_y = returns_bt.resample('YE').mean()*252

    # obtenemos los retornos anuales de cada portafolio usando los pesos de portafolios optimizados
    for i in range(1, len(returns_bt_y) + 1):
        sub_returns = returns_bt_y.iloc[:i].T

        returns_min_vol = np.dot(np.array(st.session_state.min_vol_resultados["min_vol_pesos"]).T, sub_returns)
        returns_sr = np.dot(np.array(st.session_state.max_sr_resultados["max_sr_pesos"]).T, sub_returns)
        returns_min_obj = np.dot(np.array(st.session_state.min_obj_resultados["min_obj_pesos"]).T, sub_returns)
        returns_ew = np.dot(np.array(len(returns_bt_y.columns)*[100./len(returns_bt_y.columns)]).T, sub_returns)
            
    resultados_bt_y = {"Minimun Volatility": returns_min_vol.tolist()[0],
                    "Maximum Sharpe Ratio": returns_sr.tolist()[0],
                    "Min. Vol. Target": returns_min_obj.tolist()[0],
                    "Equaly Weighted": returns_ew.tolist()}
    resultados_bt_y = pd.DataFrame.from_dict(resultados_bt_y).set_index(returns_bt_y.index.year)    
        
    # hacemos lo mismo proceso con los datos del S&P 500
    returns_sp500_y = st.session_state.returns_sp500.copy()
    returns_sp500_y.index = pd.to_datetime(returns_sp500_y.index)
    returns_sp500_y = (returns_sp500_y.resample('YE').apply(lambda x: (1 + x).prod() - 1))*100
    returns_sp500_y =returns_sp500_y.set_index(returns_sp500_y.index.year).rename(columns = {"^GSPC": "S&P 500"})

    # agregamos los datos al dataframe
    resultados_bt_y = resultados_bt_y.join(returns_sp500_y)
    # 1. graficamos los resultados anuales
    colors1 = sns.color_palette("mako", n_colors=resultados_bt_y.shape[1])
    resultados_bt_y = resultados_bt_y.reset_index()
    annual_graf_df = pd.melt(resultados_bt_y, id_vars = "Date", var_name = "Portfolio",
                                value_name= "Return")
    # visualizamos los datos
    g = sns.catplot(data = annual_graf_df, x = 'Date', y = "Return", hue = 'Portfolio',
                kind = "bar", height = 7, aspect = 2, palette=colors1, legend = True)

    # Agregar etiquetas encima de las barras
    for ax in g.axes.flat:
        for container in ax.containers:  # Recorrer cada conjunto de barras
            ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=12, color="black", padding = 3)  # Etiquetas fuera de las barras

    g.fig.suptitle("Annual Return Comparison", fontsize=18, y=1.1)
    #plt.subplots_adjust(top=0.9)
    g.set_axis_labels("", "Annual Return (%)", fontsize=15)
    ax.grid(axis = "y", linestyle = "--", alpha = 0.7)
    sns.move_legend(g, "lower center", bbox_to_anchor = (0.5, 1), ncol = 5, frameon = False, title = "")

    st.pyplot(g)
    plt.close()
    

    ## 2. Comportamiento diario
    st.subheader("Dailly Returns")
    # obtenemos los retornos anuales de cada portafolio usando los pesos de portafolios optimizados
    for i in range(1, len(returns_bt) + 1):
        sub_returns = returns_bt.iloc[:i].T
        returns_min_vol_d = np.dot(np.array(st.session_state.min_vol_resultados["min_vol_pesos"]).T, sub_returns)
        returns_sr_d = np.dot(np.array(st.session_state.max_sr_resultados["max_sr_pesos"]).T, sub_returns)
        returns_min_obj_d = np.dot(np.array(st.session_state.min_obj_resultados["min_obj_pesos"]).T, sub_returns)
        returns_ew_d = np.dot(np.array(len(returns_bt_y.columns)*[100./len(returns_bt_y.columns)]).T, sub_returns)


    resultados_bt_d = {"Minimum Volatility": (returns_min_vol_d.tolist()[0]),
                    "Maximum Sharpe Ratio": returns_sr_d.tolist()[0],
                    "Min. Vol. Target": returns_min_obj_d.tolist()[0],
                    "Equaly Weighted": returns_ew_d.tolist()}
    resultados_bt_d = pd.DataFrame.from_dict(resultados_bt_d).set_index(st.session_state.returns_bt.index) 

    # dividimos los retornos entre 100
    resultados_bt_d = resultados_bt_d / 100
    # agregamos los retornos del SP 500
    resultados_bt_d = resultados_bt_d.join(st.session_state.returns_sp500).fillna(0).rename(columns = {'^GSPC': 'S&P 500'})

    # definimos una inversion inicial en el portafolio
    inversion_inicial = 100
    # obtenemos el valor de cada portafolio a traves del tiempo
    valor_portafolio = inversion_inicial * (1 + resultados_bt_d).cumprod()

    colors2 = sns.color_palette("mako", n_colors=resultados_bt_y.shape[1]).as_hex()     
    fig = go.Figure()
    # agregamos una linea para cada portafolio
    for idx, column in enumerate(valor_portafolio.columns):
        fig.add_trace(
            go.Scatter(
                x = valor_portafolio.index,
                y = valor_portafolio[column],
                mode = "lines",
                name = column,
                line = dict(width = 2, color = colors2[idx])
            )
        )
    # estilo del grafico
    fig.update_layout(
        title = f"Daily Value of the Portfolio with Initial Investment of: ${inversion_inicial} {st.session_state.target_currency}",
        yaxis_title = f"Closing Price {st.session_state.target_currency}",
        template="plotly_white",  # Tema limpio y profesional
        hovermode="x unified",  # Tooltips unificados para todas las líneas
        width=800,
        height=600,
        legend=dict(
            title="",  # Título de la leyenda
            orientation="h",  # Posición horizontal de la leyenda
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )
    # Renderiza el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Backtesting Statistics")
                
    # observemos las estadisticas del backtesting
    if st.session_state.rf_use == "United States":
        bt_stats = st.session_state.metricas(resultados_bt_d, rf_rate = st.session_state.rf_rate_us.iloc[-1].iloc[0])
    elif st.session_state.rf_use == "Mexico":
        bt_stats = st.session_state.metricas(resultados_bt_d, rf_rate = st.session_state.rf_rate_mx.iloc[-1].iloc[0])
    st.dataframe(bt_stats)

    st.text("Great! Analyze your results and proceed to the Black-Litterman model.")

    # Botón para ir a la siguiente página
    col1, col2 = st.columns([1,0.2])
    with col2:
        if st.button("Black-Litterman!"):
                st.switch_page("pages/3_Black-Litterman Model.py")
# ax = plotting.plot_efficient_frontier(cla, showfig=False)
# ax.figure.savefig('output.png')

