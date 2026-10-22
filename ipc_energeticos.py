"""Econometría II: IPC de energéticos (DANE), enero 2010 - septiembre 2026.

Metodología Box-Jenkins: identificación, estimación y validación.
El pronóstico de 10 meses queda como etapa posterior.
Ejecutar: py ipc_energeticos.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Guarda las figuras sin abrir ventanas emergentes
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, norm, probplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning

BASE = Path(__file__).resolve().parent
SALIDA = BASE / "resultados"
SALIDA.mkdir(exist_ok=True)

# --- Datos ---------------------------------------------------------------
# Usamos la variación mensual publicada por el DANE (columna D, hoja 11).
# No confundirla con el nivel del índice ni con su primera diferencia.
meses = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}

df = pd.read_excel(
    BASE / "anex-IPC-sep2026.xlsx", sheet_name="11", usecols="A:D",
    skiprows=9, nrows=213, header=None,
    names=["anio", "mes", "indice", "variacion_mensual"],
)
df["numero_mes"] = df["mes"].astype(str).str.strip().str.lower().map(meses)
if df[["anio", "numero_mes", "variacion_mensual"]].isna().any().any():
    raise ValueError("Faltan datos o existen meses no reconocidos en el Excel.")
df["fecha"] = pd.to_datetime({
    "year": df["anio"].astype(int),
    "month": df["numero_mes"].astype(int),
    "day": 1,
})
serie = (df.set_index("fecha")["variacion_mensual"].astype(float)
         .loc["2010-01":"2026-09"].asfreq("MS"))
serie.name = "variacion_mensual_pct"
if len(serie) != 201 or serie.isna().any():
    raise ValueError("La serie debe tener 201 observaciones mensuales consecutivas.")
serie.rename_axis("fecha").to_csv(SALIDA / "serie_ipc.csv", encoding="utf-8-sig")
print(f"Serie: {serie.index.min():%Y-%m} a {serie.index.max():%Y-%m} ({len(serie)} datos)")

# --- 1. Identificación ----------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(serie.index, serie, color="#2665A2", lw=1.25)
ax.axhline(0, color="gray", ls="--", lw=0.8)
ax.set(title="Variación mensual del IPC de energéticos", xlabel="Fecha",
       ylabel="Variación mensual (%)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(SALIDA / "serie_ipc.png", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
plot_acf(serie, lags=24, alpha=0.05, bartlett_confint=False, ax=axes[0])
plot_pacf(serie, lags=24, alpha=0.05, ax=axes[1])
axes[0].set(title="FAC: variación mensual del IPC", xlabel="Rezago")
axes[1].set(title="FACP: variación mensual del IPC", xlabel="Rezago")
fig.tight_layout()
fig.savefig(SALIDA / "fac_facp_ipc.png", dpi=200)
plt.close(fig)

adf = adfuller(serie)  # H0: raíz unitaria
with warnings.catch_warnings():
    warnings.simplefilter("ignore", InterpolationWarning)
    kpss_res = kpss(serie, regression="c", nlags="auto")  # H0: estacionariedad
print(f"ADF: p={adf[1]:.6g}; KPSS: p reportado={kpss_res[1]:.3f}")
(SALIDA / "estacionariedad.txt").write_text(
    "ADF (H0: raíz unitaria)\n"
    f"Estadístico = {adf[0]:.6f}; p = {adf[1]:.9f}; rezagos = {adf[2]}\n\n"
    "KPSS (H0: estacionariedad en torno a una constante)\n"
    f"Estadístico = {kpss_res[0]:.6f}; p reportado = {kpss_res[1]:.3f}; "
    f"rezagos = {kpss_res[2]}\n"
    "Nota: si p = 0.10 por límite de la tabla, el valor p real es mayor.\n",
    encoding="utf-8",
)

# --- 2. Estimación --------------------------------------------------------
# Órdenes propuestos tras revisar la FAC/FACP (sin selección automática).
ordenes = {"AR(1)": (1, 0, 0), "AR(2)": (2, 0, 0), "ARMA(1,1)": (1, 0, 1)}
modelos = {}
comparacion = []
coeficientes = []
for nombre, orden in ordenes.items():
    modelo = SARIMAX(serie, order=orden, trend="c",
                     enforce_stationarity=False, enforce_invertibility=False)
    # Misma muestra efectiva al calcular AIC/BIC (199 observaciones).
    modelo.loglikelihood_burn = 2
    ajuste = modelo.fit(disp=False)
    if not ajuste.mle_retvals.get("converged", False):
        raise RuntimeError(f"No convergió el modelo {nombre}.")
    modelos[nombre] = ajuste
    comparacion.append({"modelo": nombre, "AIC": ajuste.aic,
                        "BIC": ajuste.bic, "log_verosimilitud": ajuste.llf})
    for parametro in ajuste.params.index:
        coeficientes.append({
            "modelo": nombre, "parametro": parametro,
            "estimacion": ajuste.params[parametro],
            "error_estandar": ajuste.bse[parametro],
            "valor_p": ajuste.pvalues[parametro],
        })

tabla = pd.DataFrame(comparacion).sort_values("AIC")
tabla.to_csv(SALIDA / "comparacion_modelos.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(coeficientes).to_csv(
    SALIDA / "coeficientes_modelos.csv", index=False, encoding="utf-8-sig")
print("\nComparación de modelos:")
print(tabla.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# El ARMA(1,1) es el candidato según AIC/BIC; revisar sus residuos.
ajuste = modelos["ARMA(1,1)"]
(SALIDA / "resumen_arma11.txt").write_text(ajuste.summary().as_text(), encoding="utf-8")

# --- 3. Validación --------------------------------------------------------
# Excluir 2 observaciones iniciales usadas para inicializar el filtro.
residuos = ajuste.resid.iloc[int(ajuste.loglikelihood_burn):].dropna()
rezagos = [6, 12, 18, 24]
ljung_box = acorr_ljungbox(residuos, lags=rezagos, model_df=2,
                           return_df=True)
ljung_box.to_csv(SALIDA / "prueba_ljung_box.csv", encoding="utf-8-sig")
arch_est, arch_p, _, _ = het_arch(residuos, nlags=12)
jb_est, jb_p = jarque_bera(residuos)
print("\nLjung-Box (p-valores):")
print(ljung_box["lb_pvalue"].round(4).to_string())
print(f"ARCH-LM: p={arch_p:.6f}; Jarque-Bera: p={jb_p:.9g}")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes[0, 0].plot(residuos.index, residuos, color="#2665A2", lw=1)
axes[0, 0].axhline(0, ls="--", color="gray")
axes[0, 0].set(title="Residuos", ylabel="Puntos porcentuales")
plot_acf(residuos, lags=24, alpha=0.05, zero=False, ax=axes[0, 1])
axes[0, 1].set_title("FAC de los residuos")
axes[1, 0].hist(residuos, bins=20, density=True, color="#2665A2", alpha=.7)
x = np.linspace(residuos.min(), residuos.max(), 200)
axes[1, 0].plot(x, norm.pdf(x, residuos.mean(), residuos.std(ddof=1)),
                color="#C14953", lw=2)
axes[1, 0].set_title("Distribución de los residuos")
probplot(residuos, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Gráfico Q-Q")
fig.tight_layout()
fig.savefig(SALIDA / "diagnostico_residuos.png", dpi=200)
plt.close(fig)

(SALIDA / "validacion.txt").write_text(
    "Validación del ARMA(1,1)\n\n"
    "Ljung-Box (H0: no hay autocorrelación hasta el rezago indicado):\n"
    + ljung_box.round(5).to_string()
    + f"\n\nARCH-LM (12 rezagos): estadístico={arch_est:.4f}; p={arch_p:.6f}\n"
    + f"Jarque-Bera: estadístico={jb_est:.4f}; p={jb_p:.9g}\n"
    + "Nota: la falta de normalidad aconseja cautela al construir intervalos de pronóstico gaussianos.\n",
    encoding="utf-8",
)
print("\nResultados guardados en la carpeta 'resultados'.")

# --- 4. Pronóstico --------------------------------------------------------
# Esta etapa queda para el integrante encargado del pronóstico.
# Utilizar el objeto `ajuste`, que contiene el ARMA(1,1) estimado:
# futuro = ajuste.get_forecast(steps=10)
# predicciones = futuro.predicted_mean
# intervalos_95 = futuro.conf_int(alpha=0.05)
# La variable pronosticada es la variación mensual porcentual del IPC,
# NO el nivel original del índice. Interpretar con cautela los intervalos.
