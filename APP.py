## I N T E G R A N T E S : ##
## García Acevedo Víctor Manuel 421095874 ##
## García Hernández Karla Giovana 318316051 ##
## Hernández Mata Verónica 422110088 ##
## Tavani Guzmán Camilla 319049211 ##

import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import plotly.graph_objects as go
import plotly.express as px
import scipy.optimize as sco
import streamlit as st
from datetime import date
from datetime import datetime
from streamlit_option_menu import option_menu  # Para un menú de navegación atractivo

## INCISO A ##
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


# Función principal para análisis
def analizar_etfs(etfs, start_date, end_date):
    data = obtener_datos_acciones(etfs, start_date, end_date)  # Descargar los precios
    resultados = {}
    rendimientos_historicos = pd.DataFrame()
    rendimientos_esperados = pd.DataFrame()

    for etf in etfs:
        # Calcular métricas estadísticas individuales
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

        # Calculamos rendimientos históricos
        rendimientos_historicos[etf] = data[etf].pct_change().dropna()
        # Rendimientos esperados como el promedio de los rendimientos históricos
        rendimientos_esperados.at[0, etf] = data[etf].pct_change().mean()

    # Convertimos los resultados a un DataFrame
    df_resultados = pd.DataFrame(resultados).T  # Transponemos para que cada ETF sea una fila
    return df_resultados, rendimientos_historicos, rendimientos_esperados



# Configuración inicial de la página
st.set_page_config(
    page_title="Análisis de ETFs",
    page_icon="📊",
    layout="wide"
)

# Encabezado
st.title("📊 Análisis Avanzado de ETFs")
st.markdown("""Explora y analiza las principales métricas estadísticas de los ETFs seleccionados.""")

# Sidebar con menú de navegación
with st.sidebar:
    selected = option_menu(
        "Menú",
        ["Introducción", "Análisis Estadístico", "Gráfica de Precios", "Descripción de ETFs Considerados","Optimización de Portafolios", "Backtesting", "Analisis Backtesting","Modelo Black-Litterman","Conclusiones Modelo Black-Litterman"],
        icons=["house", "bar-chart-line", "line-chart", "gear", "info-circle"],
        menu_icon="menu-up",
        default_index=0
    )

# Lógica según el menú
if selected == "Introducción":
    st.header("Bienvenido")
    st.write("""
    Esta herramienta utiliza datos históricos para calcular métricas avanzadas de ETFs, optimizar portafolios,
    y simular la frontera eficiente.
    """)

elif selected == "Análisis Estadístico":
    st.header("🔍 Análisis Estadístico de ETFs")
    etfs = ["LQD", "EMB", "VTI", "EEM", "GLD"]  # Símbolos de los ETFs disponibles
    selected_etfs = st.multiselect("Selecciona los ETFs para analizar", etfs, default=etfs)
    start_date = st.date_input("Fecha de inicio", datetime(2010, 1, 1))
    end_date = st.date_input("Fecha de fin", datetime(2023, 12, 31))

    # Botón para ejecutar el análisis
    if st.button("Calcular métricas"):
        df_resultados, rendimientos_historicos, rendimientos_esperados = analizar_etfs(
            selected_etfs, start_date, end_date
        )

        # Mostrar los resultados
        st.subheader("📊 Resultados del análisis:")
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
        
        # Mostrar rendimientos esperados
        st.subheader("📈 Rendimientos esperados por ETF:")
        st.dataframe(rendimientos_esperados)

        # Mostrar rendimientos históricos
        st.subheader("📊 Rendimientos históricos diarios:")
        st.dataframe(rendimientos_historicos)

elif selected == "Gráfica de Precios":
    st.header("📈 Gráfica de Precios de los ETFs")
    etfs = ["LQD", "EMB", "VTI", "EEM", "GLD"]
    selected_etfs = st.multiselect("Selecciona los ETFs para graficar", etfs, default=["LQD", "VTI"])

    # Fechas predeterminadas
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2023, 12, 31)

    # Descargar datos
    data = obtener_datos_acciones(selected_etfs, start_date, end_date)

    # Gráfico
    fig = go.Figure()
    for etf in selected_etfs:
        fig.add_trace(go.Scatter(x=data.index, y=data[etf], mode='lines', name=etf))

    fig.update_layout(title='Precios Históricos de los ETFs',
                      xaxis_title='Fecha',
                      yaxis_title='Precio de Cierre')
    st.plotly_chart(fig)

elif selected == "Descripción de ETFs Considerados":
    st.header("📝 Descripción de ETFs Considerados")
    st.markdown("""

### 1. LQD - iShares iBoxx $ Investment Grade Corporate Bond ETF

**Exposición que gana:**  
El LQD ofrece exposición a bonos corporativos de grado de inversión denominados en dólares estadounidenses. Su estrategia se centra en invertir en instrumentos de deuda emitidos por algunas de las empresas más grandes y establecidas, principalmente en los Estados Unidos. Es un fondo diseñado para inversores que buscan un rendimiento moderado y un perfil de riesgo relativamente bajo, con la seguridad de invertir en deuda corporativa de alta calidad.

**Índice que sigue:**  
Este ETF sigue el *iBoxx $ Liquid Investment Grade Index*, que está compuesto por bonos corporativos emitidos en dólares por empresas de grado de inversión. Este índice incluye bonos de emisores altamente calificados, con una calificación crediticia mínima de BBB- por parte de agencias como S&P o Fitch.

**Moneda de denominación:**  
Denominado en USD (dólares estadounidenses), lo que le permite acceder a los mercados de bonos corporativos de EE.UU. con una exposición directa a la moneda más utilizada a nivel global.

**Estilo de inversión:**  
El LQD se clasifica dentro del estilo de *grado de inversión* en renta fija, con un enfoque conservador y de preservación de capital.

**Principales contribuyentes:**  
El fondo invierte en una amplia gama de bonos emitidos por algunas de las empresas más grandes y estables de Estados Unidos, como Apple, Microsoft, Johnson & Johnson, ExxonMobil y Berkshire Hathaway.

**Países donde invierte:**  
Principalmente Estados Unidos, con una menor exposición a emisores en otros países que estén denominados en dólares.

**Métricas de riesgo:**  
- **Duración:** Aproximadamente 7 años.  
- **Beta:** Entre 0.2 - 0.4.  
- **Rendimiento esperado:** Entre 3.5% y 4%.  

**Costos:**  
- **Ratio de gastos:** 0.14%.  

---

### 2. EMB - iShares JP Morgan USD Emerging Markets Bond ETF

**Exposición que gana:**  
El EMB ofrece exposición a bonos soberanos y corporativos denominados en dólares de países emergentes. Está diseñado para inversores que buscan diversificar su portafolio con activos de renta fija fuera de los mercados desarrollados, pero que buscan mitigar el riesgo cambiario a través de la denominación en dólares.

**Índice que sigue:**  
Este ETF sigue el *J.P. Morgan EMBI Global Index*, el cual incluye bonos de deuda emitidos por gobiernos y empresas de mercados emergentes, denominados en dólares estadounidenses.

**Moneda de denominación:**  
Denominado en USD, eliminando el riesgo de cambio asociado con las monedas locales.

**Estilo de inversión:**  
Se considera un fondo de renta fija de alto riesgo, con un enfoque en la estrategia de *alto rendimiento*.

**Principales contribuyentes:**  
Bonos de países como México, Brasil, Sudáfrica, Indonesia e India.

**Países donde invierte:**  
México, Brasil, Sudáfrica, Indonesia, India, entre otros.

**Métricas de riesgo:**  
- **Duración:** Entre 5 y 7 años.  
- **Beta:** Entre 0.6 y 0.8.  
- **Rendimiento esperado:** Entre 4.5% y 5.5%.  

**Costos:**  
- **Ratio de gastos:** 0.39%.  

---

### 3. VTI - Vanguard Total Stock Market ETF

**Exposición que gana:**  
El VTI ofrece exposición a una cartera diversificada de acciones de EE.UU., que cubren el mercado completo de renta variable, desde grandes empresas hasta empresas de pequeña capitalización.

**Índice que sigue:**  
Este ETF sigue el *CRSP US Total Market Index*, que incluye todas las acciones cotizadas en los EE.UU.

**Moneda de denominación:**  
Denominado en USD, exclusivamente en empresas estadounidenses.

**Estilo de inversión:**  
Se clasifica dentro del estilo *growth*, con una alta concentración en tecnología y consumo discrecional.

**Principales contribuyentes:**  
Apple, Microsoft, Amazon y Tesla son las principales acciones en el fondo.

**Países donde invierte:**  
Estados Unidos.

**Métricas de riesgo:**  
- **Beta:** 1 (similar a la volatilidad del mercado general).  
- **Rendimiento esperado:** Entre 7% y 10%.  

**Costos:**  
- **Ratio de gastos:** 0.03%.

---

### 4. EEM - iShares MSCI Emerging Markets ETF

**Exposición que gana:**  
El EEM ofrece exposición a acciones de mercados emergentes, capturando el crecimiento de economías en desarrollo con mayor potencial de expansión en el futuro.

**Índice que sigue:**  
Este ETF sigue el *MSCI Emerging Markets Index*, que incluye más de 800 empresas en 26 países emergentes.

**Moneda de denominación:**  
Denominado en USD, lo que elimina la exposición a monedas locales.

**Estilo de inversión:**  
Se ajusta al estilo *growth*, con énfasis en sectores como tecnología y consumo.

**Principales contribuyentes:**  
China, Corea del Sur, Taiwán, India, Brasil y empresas como Tencent, Samsung y Alibaba.

**Países donde invierte:**  
China, India, Brasil, Corea del Sur, Taiwán.

**Métricas de riesgo:**  
- **Beta:** Mayor a 1, indicando una volatilidad más alta comparada con activos desarrollados.  
- **Rendimiento esperado:** Entre 6% y 8%.  

**Costos:**  
- **Ratio de gastos:** 0.68%.

---

### 5. GLD - SPDR Gold Shares ETF

**Exposición que gana:**  
El GLD ofrece exposición al oro físico, permitiendo a los inversores beneficiarse de la subida del precio del oro sin tener que poseer físicamente el metal.

**Índice que sigue:**  
No sigue un índice específico, simplemente refleja el precio del oro físico en el mercado internacional.

**Moneda de denominación:**  
Denominado en USD, siguiendo el mercado global del oro.

**Estilo de inversión:**  
Se considera un activo de *cobertura* y *refugio seguro* en tiempos de crisis e inflación.

**Principales contribuyentes:**  
Posee oro físico almacenado en bóvedas.

**Países donde invierte:**  
No tiene una exposición geográfica específica.

**Métricas de riesgo:**  
- **Beta:** Cercana a 0.  
- **Rendimiento esperado:** Entre 3% y 5%.

**Costos:**  
- **Ratio de gastos:** 0.40%.


    """)
elif selected == "Optimización de Portafolios":
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
elif selected == "Backtesting":
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
elif selected == "Modelo Black-Litterman":
    import streamlit as st
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Título de la aplicación
    st.title("Optimización de Portafolio - Modelo Black-Litterman")

    # 1. Descargar datos históricos
    assets = ['LQD', 'EMB', 'VTI', 'EEM', 'GLD']
    start_date = '2020-01-01'
    end_date = '2023-12-31'


    # Descargar precios de cierre ajustados
    st.write("Descargando datos históricos de activos...")
    prices = yf.download(assets, start=start_date, end=end_date)['Adj Close']

    # Calcular rendimientos diarios y anuales
    daily_returns = prices.pct_change().dropna()
    annual_returns = daily_returns.mean() * 252  # 252 días de mercado en un año

    st.write("Rendimientos esperados anuales:")
    st.write(annual_returns)

    # Matriz de covarianzas anual
    cov_matrix = daily_returns.cov() * 252

    # 3. Pesos equitativos
    num_assets = len(assets)
    equal_weights = np.ones(num_assets) / num_assets

    # Rendimientos esperados
    risk_aversion = 2.5  # Aversión al riesgo para el mercado
    market_implied_returns = risk_aversion * np.dot(cov_matrix, equal_weights)

    st.write("Rendimientos implícitos del mercado:")
    st.write(market_implied_returns)

    # 4. Opiniones del inversionista
    P = np.array([
        [0, 1, -1, 0, 0],  # EMB > VTI
        [0, 0, 0, 1, -1]   # EEM > GLD
    ])
    Q = np.array([0.005, 0.01])  # Opiniones expresadas como diferencias de rendimiento
    tau = 0.025
    omega = np.diag(np.dot(np.dot(P, cov_matrix), P.T).diagonal() * tau)  # Crear matriz Omega

    # Cálculo de los rendimientos ajustados con la metodología Black-Litterman
    adjusted_returns = np.linalg.inv(
        np.linalg.inv(tau * cov_matrix) + np.dot(np.dot(P.T, np.linalg.inv(omega)), P)
    ).dot(
        np.dot(np.linalg.inv(tau * cov_matrix), market_implied_returns) +
        np.dot(np.dot(P.T, np.linalg.inv(omega)), Q)
    )

    # Matriz de covarianza ajustada
    adjusted_cov_matrix = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(np.dot(P.T, np.linalg.inv(omega)), P))

    st.write("Rendimientos ajustados (Black-Litterman):")
    st.write(adjusted_returns)

    # Optimización del portafolio
    optimal_weights = np.linalg.inv(risk_aversion * adjusted_cov_matrix).dot(adjusted_returns)
    optimal_weights = optimal_weights / np.sum(optimal_weights)  # Normalizamos los pesos

    st.write("Pesos óptimos del portafolio:")
    portfolio_df = pd.DataFrame({
        'Asset': assets,
        'Weight': optimal_weights
    })
    st.write(portfolio_df)

    # Gráfica de barras con los resultados
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Asset', y='Weight', data=portfolio_df, palette="viridis", ax=ax)
    ax.set_title('Pesos Optimizados del Portafolio (Black-Litterman)', fontsize=16)
    ax.set_xlabel('Activos', fontsize=14)
    ax.set_ylabel('Pesos', fontsize=14)
    ax.set_ylim(0, max(optimal_weights) + 0.05)
    st.pyplot(fig)  # Mostrar gráfica en Streamlit

    # Gráfica circular
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.pie(
        portfolio_df['Weight'],
        labels=portfolio_df['Asset'],
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette("viridis", len(portfolio_df))
    )
    ax2.set_title('Distribución de Pesos del Portafolio')
    st.pyplot(fig2)  # Mostrar gráfica circular
elif selected == "Analisis Backtesting":
        st.header("📝 Backtesting de los Portafolios Óptimos (2021-2023)")
        st.markdown("""

Con base en los resultados de las optimizaciones del portafolio, hemos evaluado el rendimiento de tres estrategias de inversión en el período comprendido entre 2021 y 2023: un portafolio con el máximo Sharpe Ratio, uno con la mínima volatilidad, y uno con pesos equitativos. Además, comparamos estos resultados con el rendimiento del S&P 500 durante el mismo período. A continuación se presentan las métricas calculadas para cada uno de los portafolios evaluados, así como una comparación entre ellos.

---

## 1. Portafolio con el Máximo Sharpe Ratio

- **Rendimiento Anualizado:** -3.79%  
- **Volatilidad Anualizada:** 71.52%  
- **Skewness:** -0.43  
- **Kurtosis:** 0.06  
- **VaR (95%):** -9.80%  
- **CVaR (95%):** -12.05%  
- **Sharpe Ratio:** -5.30  
- **Sortino Ratio:** -112.53  
- **Max Drawdown:** -100%  

El portafolio con el máximo Sharpe Ratio tuvo un rendimiento anualizado negativo de -3.79%, con una volatilidad extremadamente alta (71.52%). Además, presentó una pérdida masiva en el Max Drawdown, cayendo un 100%, lo que indica una caída total de valor en el peor momento.  

El Sharpe Ratio es negativo (-5.30), lo que refleja una mala compensación entre riesgo y retorno. Asimismo, el Sortino Ratio, que es más sensible a los retornos negativos, también es extremadamente negativo (-112.53), lo que refuerza la mala performance del portafolio.

---

## 2. Portafolio con Mínima Volatilidad

- **Rendimiento Anualizado:** -22.34%  
- **Volatilidad Anualizada:** 160.90%  
- **Skewness:** 0.25  
- **Kurtosis:** -1.46  
- **VaR (95%):** -20.97%  
- **CVaR (95%):** -24.83%  
- **Sharpe Ratio:** -13.88  
- **Sortino Ratio:** -373.57  
- **Max Drawdown:** -100%  

El portafolio con mínima volatilidad presenta un rendimiento muy bajo de -22.34% anualizado y una volatilidad extremadamente alta (160.90%), que refleja un riesgo excesivo en relación con el rendimiento.  

Además, el Sharpe Ratio (-13.88) y el Sortino Ratio (-373.57) muestran que este portafolio no es eficiente en términos de la relación entre el riesgo y el retorno, con grandes caídas que resultan en pérdidas significativas.  

El Max Drawdown también alcanzó un 100%, lo que indica una pérdida completa de valor en el peor momento.

---

## 3. Portafolio con Pesos Equitativos

- **Rendimiento Anualizado:** -20.41%  
- **Volatilidad Anualizada:** 115.17%  
- **Skewness:** 0.05  
- **Kurtosis:** -1.36  
- **VaR (95%):** -17.78%  
- **CVaR (95%):** -21.16%  
- **Sharpe Ratio:** -17.73  
- **Sortino Ratio:** -358.15  
- **Max Drawdown:** -100%  

El portafolio equitativo también experimentó un rendimiento negativo, con una pérdida anualizada de -20.41% y una volatilidad de 115.17%.  

A pesar de la distribución uniforme entre los activos, el Sharpe Ratio y el Sortino Ratio siguen siendo negativos, lo que indica que no hubo una relación favorable entre el riesgo y el retorno.  

Además, al igual que en los portafolios anteriores, el Max Drawdown fue de 100%, sugiriendo una caída total en el peor escenario.

---

## 4. Rendimiento del S&P 500

- **Rendimiento Anualizado:** 35.27%  
- **Volatilidad Anualizada:** 121.76%  
- **Skewness:** -0.04  
- **Kurtosis:** -0.93  
- **VaR (95%):** 2.13%  
- **CVaR (95%):** -0.01%  
- **Sharpe Ratio:** 28.96  
- **Sortino Ratio:** 3349.67  
- **Max Drawdown:** -16.30%  

En comparación con los portafolios optimizados, el S&P 500 presentó un rendimiento excepcional de 35.27% anualizado, con una volatilidad de 121.76%.  

Aunque su volatilidad fue elevada, el Sharpe Ratio (28.96) y el Sortino Ratio (3349.67) destacan positivamente, indicando que el rendimiento fue favorable en relación con el riesgo asumido.  

El Max Drawdown fue de solo -16.30%, mucho más bajo en comparación con los otros portafolios, lo que indica una pérdida mucho más controlada en el peor escenario.

---

## Conclusión: ¿Dónde Hubiera Sido Mejor Invertir?

A pesar de que los portafolios optimizados con el máximo Sharpe Ratio y la mínima volatilidad estaban diseñados para maximizar la relación entre riesgo y retorno, ambos portafolios resultaron ser deficientes en términos de rendimiento, con pérdidas significativas a lo largo del período analizado.  

Los portafolios presentaron rendimientos anuales negativos de alrededor de:  

- **-3.79%** para el máximo Sharpe Ratio.  
- **-22.34%** para el mínimo nivel de volatilidad.  

Con caídas extremas que llevaron a pérdidas de valor cercanas al 100%.

En contraste, el S&P 500 superó ampliamente a los portafolios optimizados con un rendimiento anualizado de **35.27%**, un Sharpe Ratio de **28.96**, y un Max Drawdown mucho más moderado de **-16.30%**. Esto demuestra que, durante el período de 2021 a 2023, el mercado estadounidense fue mucho más rentable y menos arriesgado que las combinaciones de ETF seleccionados.

El portafolio equitativo, aunque con una distribución balanceada entre los activos, no logró superar el rendimiento del S&P 500, obteniendo un rendimiento anualizado de **-20.41%** y con un alto nivel de volatilidad.

---

## Conclusión Final

Si hubiéramos invertido nuestros recursos en el S&P 500, habríamos obtenido un rendimiento significativamente superior al de cualquier portafolio optimizado durante este período.  

A pesar de las caídas que enfrentó el mercado en algunos momentos, la diversificación inherente al S&P 500 y su recuperación constante hicieron que fuera la mejor opción.  

El análisis muestra que, en este caso específico, el enfoque de invertir en un índice de mercado amplio resultó ser la estrategia más efectiva, con un rendimiento estable y una mayor gestión del riesgo en comparación con los portafolios de ETF seleccionados.
    """)
elif selected == "Conclusiones Modelo Black-Litterman":
        st.header("📝 Análisis de Rendimiento de Activos Financieros")
        st.markdown("""El rendimiento de los activos financieros está influenciado por factores globales, como:

- **Inflación persistente:** Los bancos centrales (como la Fed) están moderando aumentos de tasas de interés.
- **Recesión técnica en algunas regiones:** Impulsa activos refugio como el oro.
- **Recuperación económica desigual:** Mercados emergentes muestran oportunidades debido a mayores proyecciones de crecimiento en Asia.
- **Condiciones del mercado de bonos:** Inversionistas buscan rendimientos reales positivos en deuda corporativa y emergente.

---

## Rendimiento Justificado de los Activos

| Activo | Rendimiento Justificado | Razón |
|--------|--------------------------|-------|
| **LQD** | 0.10% | Bonos corporativos aún afectados por tasas altas; limitado crecimiento en precio. |
| **EMB** | -0.95% | Alta deuda en mercados emergentes, pero mejora relativa en recuperación económica. |
| **VTI** | 13.40% | Recuperación en sectores tecnológicos tras ajuste de tasas en EE.UU. |
| **EEM** | 1.74% | Impulso por China y Asia; volatilidad política y monetaria restringen el rendimiento. |
| **GLD** | 8.34% | Aumento de demanda por refugio frente a incertidumbre geopolítica e inflación persistente. |

---

## Descripción de los Activos

- **LQD:** *iShares iBoxx $ Investment Grade Corporate Bond ETF.*  
  Este ETF invierte en bonos corporativos de grado de inversión, lo que implica un menor riesgo de default pero también menores rendimientos potenciales.

- **EMB:** *JPMorgan Emerging Markets Bond Index ETF.*  
  Este ETF invierte en bonos de mercados emergentes, ofreciendo mayores rendimientos potenciales pero también mayor volatilidad y riesgo.

- **VTI:** *Vanguard Total Stock Market Index Fund ETF.*  
  Este ETF proporciona una exposición amplia al mercado de acciones estadounidense.

- **EEM:** *iShares MSCI Emerging Markets ETF.*  
  Este ETF ofrece exposición a las acciones de grandes y medianas capitalizaciones en mercados emergentes.

- **GLD:** *SPDR Gold Shares.*  
  Este ETF rastrea el precio del oro, considerado a menudo como un activo de refugio y una cobertura contra la inflación.

---

## Posibilidades Económicas y Justificación

### Desaceleración económica global:
Se anticipa un crecimiento económico más moderado debido a factores como la política monetaria restrictiva, la inflación persistente y las tensiones geopolíticas.

### Fortalecimiento del dólar:
Un dólar más fuerte podría presionar a la baja los precios de las commodities y las acciones de mercados emergentes.

### Aumento de la volatilidad:
Se espera un entorno de mercado más volátil debido a la incertidumbre económica y geopolítica.

---

## Análisis Justificado por Activo

| Activo | Perspectiva | Justificación | Rendimiento esperado |
|--------|-------------|---------------|---------------------|
| **LQD** | Moderada | Se espera que las tasas de interés aumenten, lo que podría reducir el atractivo de los bonos. Sin embargo, la calidad crediticia de estos bonos proporciona cierta estabilidad en un entorno económico incierto. | 2-3% |
| **EMB** | Negativa | El fortalecimiento del dólar y la mayor aversión al riesgo en los mercados emergentes podrían presionar a la baja los rendimientos de estos bonos. Además, la volatilidad y el riesgo político en estos mercados aumentan la incertidumbre. | 1-2% |
| **VTI** | Moderada | Aunque el mercado de acciones estadounidense está relativamente bien valorado, la desaceleración económica global y las políticas monetarias restrictivas podrían limitar el crecimiento. Sin embargo, la diversificación amplia de este ETF proporciona cierta resiliencia. | 4-5% |
| **EEM** | Negativa | Los mercados emergentes son más vulnerables a las perturbaciones globales y al fortalecimiento del dólar. La volatilidad y los riesgos geopolíticos también afectan negativamente a estos mercados. | 2-3% |
| **GLD** | Positiva | El oro suele apreciarse en entornos de incertidumbre económica y como cobertura contra la inflación. Con la volatilidad esperada y las tensiones geopolíticas, el oro podría ser un refugio seguro. | 5-6% |

---

## Distribución a Prior

Para calcular la distribución a priori, puedes asumir un benchmark constituido por los activos seleccionados con un peso equitativo. Esto significa que cada activo tendría un peso del **20%** en el portafolio inicial.

---

## Resumen de Perspectivas de los Activos

- **LQD:** Rendimiento moderado de 2-3%.  
- **EMB:** Rendimiento bajo de 1-2%.  
- **VTI:** Rendimiento moderado de 4-5%.  
- **EEM:** Rendimiento bajo de 2-3%.  
- **GLD:** Rendimiento positivo de 5-6%.  

---

## Factores Claves a Monitorear

1. **Inflación persistente.**
2. **Cambios en políticas monetarias.**
3. **Recuperación económica desigual en diversas regiones.**
4. **Volatilidad política en mercados emergentes.**
5. **Comportamiento del oro como activo refugio.***
    """)
    
