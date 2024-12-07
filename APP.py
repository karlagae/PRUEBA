import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ------------------------------
# Datos iniciales y benchmark
# ------------------------------
# Supongamos que tienes los rendimientos históricos de tus activos seleccionados
# (renombrados como activos A, B, C, etc.)
rendimientos_historicos = pd.DataFrame({
    "Activo_A": np.random.normal(0.02, 0.05, 252),
    "Activo_B": np.random.normal(0.015, 0.04, 252),
    "Activo_C": np.random.normal(0.01, 0.03, 252)
})

# Cálculo de la media y covarianza histórica
rendimientos_medios = rendimientos_historicos.mean()
matriz_covarianza = rendimientos_historicos.cov()

# Definir un benchmark con pesos equitativos
num_activos = rendimientos_historicos.shape[1]
pesos_benchmark = np.ones(num_activos) / num_activos
rendimiento_benchmark = np.dot(pesos_benchmark, rendimientos_medios)

# ------------------------------
# Views (perspectivas del analista)
# ------------------------------
# Ejemplo: Perspectiva relativa y absoluta
# Aquí debes justificar los views en 5 bullets (agregar en comentarios).

# Views relativos: activo B sobreperforma a activo C por 1% adicional
Q = np.array([0.01])  # Rendimiento esperado adicional para el view relativo

# Matriz P: relación entre activos para el view
P = np.array([
    [0, 1, -1]  # Relación entre activo B y activo C
])

# ------------------------------
# Parámetros del modelo Black-Litterman
# ------------------------------
# Calcular la distribución a priori
tau = 0.05  # Escalar de incertidumbre del mercado
prior_cov = tau * matriz_covarianza
omega = np.dot(np.dot(P, prior_cov), P.T)  # Incertidumbre en los views

# Calcular los rendimientos esperados ajustados (posteriores)
inv_cov_prior = np.linalg.inv(prior_cov)
inv_omega = np.linalg.inv(omega)

# Fórmula Black-Litterman
posterior_mean = np.dot(
    np.linalg.inv(inv_cov_prior + np.dot(P.T, inv_omega).dot(P)),
    (inv_cov_prior.dot(rendimientos_medios) + np.dot(P.T, inv_omega).dot(Q))
)

posterior_cov = np.linalg.inv(inv_cov_prior + np.dot(P.T, inv_omega).dot(P))

# ------------------------------
# Optimización del portafolio
# ------------------------------
# Función objetivo (maximizar el ratio Sharpe)
def objetivo(pesos, rendimientos, covarianza, aversion_riesgo=2):
    rendimiento = np.dot(pesos, rendimientos)
    riesgo = np.dot(pesos.T, np.dot(covarianza, pesos))
    utilidad = rendimiento - (aversion_riesgo / 2) * riesgo
    return -utilidad  # Minimizar utilidad negativa

# Restricciones: suma de pesos = 1, no ventas en corto
restricciones = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
limites = tuple((0, 1) for _ in range(num_activos))

# Optimización
pesos_iniciales = np.ones(num_activos) / num_activos
result = minimize(
    objetivo, pesos_iniciales, args=(posterior_mean, posterior_cov),
    method='SLSQP', bounds=limites, constraints=restricciones
)

# Pesos óptimos
pesos_optimos = result.x

# ------------------------------
# Resultados finales
# ------------------------------
print("Rendimientos esperados ajustados:", posterior_mean)
print("Pesos óptimos del portafolio:", pesos_optimos)
