
from __future__ import annotations

import hashlib
import io
import re
import subprocess
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import statsmodels.api as sm
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    acorr_ljungbox,
    breaks_cusumolsresid,
    het_breuschpagan,
    linear_reset,
)
from statsmodels.stats.outliers_influence import (
    OLSInfluence,
    variance_inflation_factor,
)
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen


st.set_page_config(
    page_title="Observatorio Pasto–Nariño",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg,#0B1220,#111827);
    color: #E2E8F0;
}
section[data-testid="stSidebar"] {
    background: #0F172A;
    border-right: 1px solid #334155;
}
h1,h2,h3 {
    color: #67E8F9 !important;
}
[data-testid="stMetric"] {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 12px;
}
[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 8px;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg,#0E7490,#6D28D9);
    border: none;
}
</style>
""", unsafe_allow_html=True)

STYLE = dict(
    template="plotly_dark",
    paper_bgcolor="#0B1220",
    plot_bgcolor="#111827",
)

BASE = Path(__file__).resolve().parent
RUNTIME = BASE / "runtime"

REPO_RNDC = "JuanDavidgm2026/fletes-precios-inflacion-pasto"
REPO_SIPSA = "JuanDavidgm2026/SIPSA-PASTO"
TAG = "sipsa-dane-originales-v1"

Y = "BRECHA_INFLACION_PASTO_COLOMBIA_PP"
T = "VAR_FLETE_TORNQVIST_PCT"
P = "PM_ponderado_t"
I = "ICTC_MENSUAL_PCT"

MES_84 = pd.period_range("2018-01", "2024-12", freq="M")
MES_59 = pd.period_range("2020-02", "2024-12", freq="M")
MES_56 = pd.period_range("2020-05", "2024-12", freq="M")

NOMBRES = {
    Y: "Brecha Pasto–Colombia (p. p.)",
    T: "Fletes RNDC Törnqvist (%)",
    P: "SIPSA ponderado El Potrerillo (%)",
    I: "ICTC nacional (%)",
    "PM_no_ponderado_t": "SIPSA no ponderado (%)",
    "PM_ponderado_lagQ_t": "SIPSA pesos rezagados (%)",
    "PM_100": "SIPSA canasta completa (%)",
    "PM_95": "SIPSA canasta 95 % (%)",
    "VAR_FLETE_FISHER_PCT": "Fletes RNDC Fisher (%)",
    "VAR_FLETE_CRUDO_PCT": "Fletes RNDC sin filtro (%)",
    "VAR_FLETE_SIN_2000_PCT": "Fletes RNDC sin 2.000 (%)",
    "IPC_MENSUAL_PASTO_PCT": "Inflación mensual Pasto (%)",
    "IPC_MENSUAL_NACIONAL_PCT": "Inflación mensual nacional (%)",
    "cobertura_indice_pct": "Cobertura SIPSA (%)",
    "n_productos_indice": "Productos SIPSA emparejados",
}

HASHES = {
    "RNDC": (
        "aaec180a88d32bbccbfb13b94be4784c9e9b4c6f23d9a93b7475b1acd1bd3891"
    ),
    "IPC": (
        "3e584414f3f89c78f99ce92576a0715d3d51fac5d6dcd75addf84065e4cd4076"
    ),
    "ICTC": (
        "6fe232548e47dbf9801b75c620209f579ebd0faa2a46fc30a9a305453878205a"
    ),
}

URLS = {
    "RNDC": [
        f"https://raw.githubusercontent.com/{REPO_RNDC}/main/data/"
        "variable2_flete_regional_modelo_2020_2024.csv"
    ],
    "IPC": [
        "https://www.dane.gov.co/files/operaciones/IPC/"
        "dic2024/anex-IPC-CiudadesMensuales-dic2024.xlsx",
        f"https://github.com/{REPO_SIPSA}/releases/download/"
        f"{TAG}/anex-IPC-CiudadesMensuales-dic2024.xlsx",
    ],
    "ICTC": [
        "https://www.dane.gov.co/files/operaciones/ICTC/"
        "anex-ICTC-HistVariaContrib-dic2024.xlsx",
        f"https://github.com/{REPO_SIPSA}/releases/download/"
        f"{TAG}/anex-ICTC-HistVariaContrib-dic2024.xlsx",
    ],
}

NOTEBOOK_ID = "1Tu3ZzP1L8XlUejqbS_0eXgZ8QszqiMr7"

SIPSA_CELDA_SHA = (
    "be21823d6980f6b0f092fd58ff808fa6056c0a550f01606ec9fb1d85276da98f"
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def descargar(url, timeout=75):
    respuesta = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Observatorio-Pasto/2.1"},
    )
    respuesta.raise_for_status()
    return respuesta.content


def verificar_descarga(nombre):
    errores = []

    for url in URLS[nombre]:
        try:
            raw = descargar(url)

            if sha(raw) != HASHES[nombre]:
                raise ValueError(
                    "SHA-256 distinto del registrado en el Colab"
                )

            return raw, url

        except Exception as exc:
            errores.append(f"{url}: {str(exc)[:100]}")

    raise RuntimeError(" | ".join(errores))


def normalizar(valor):
    if pd.isna(valor):
        return ""

    texto = unicodedata.normalize(
        "NFKD",
        str(valor).strip().lower(),
    )

    texto = "".join(
        c for c in texto
        if not unicodedata.combining(c)
    )

    return " ".join(texto.split())


def numero(v):
    if pd.isna(v):
        return np.nan

    if isinstance(v, (float, int, np.number)):
        return float(v)

    valor = str(v).strip().replace("%", "").replace(",", ".")

    return float(
        pd.to_numeric(valor, errors="coerce")
    )


def fechas_mensuales(serie):
    if isinstance(serie.dtype, pd.PeriodDtype):
        return serie.dt.asfreq("M")

    texto = serie.astype("string").str.strip()

    if texto.dropna().str.fullmatch(r"\d{6}").all():
        return pd.to_datetime(
            texto,
            format="%Y%m",
            errors="coerce",
        ).dt.to_period("M")

    return pd.to_datetime(
        serie,
        errors="coerce",
    ).dt.to_period("M")


def auditar_meses(df, meses, columnas):
    if df["mes"].tolist() != list(meses):
        raise ValueError(
            f"Calendario incompleto o duplicado: "
            f"{meses[0]} a {meses[-1]}"
        )

    valores = df[columnas].to_numpy(float)

    if not np.isfinite(valores).all():
        raise ValueError(
            "Valores no observados o no finitos. No se imputan."
        )


def normalizar_base(df, nombre):
    d = df.copy()

    d.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in d.columns
    ]

    fecha = next(
        (
            c for c in [
                "mes", "FECHA", "fecha",
                "PERIODO", "periodo"
            ]
            if c in d
        ),
        None,
    )

    if fecha is None:
        raise ValueError(
            f"{nombre}: falta columna temporal reconocible"
        )

    d["mes"] = fechas_mensuales(d[fecha])

    d = d[
        d["mes"].isin(MES_84)
    ].sort_values("mes")

    if d["mes"].duplicated().any():
        raise ValueError(
            f"{nombre}: hay meses duplicados"
        )

    columnas = [
        c for c in NOMBRES
        if c in d
    ]

    if not columnas:
        raise ValueError(
            f"{nombre}: ninguna columna del estudio identificada"
        )

    for c in columnas:
        d[c] = pd.to_numeric(
            d[c],
            errors="coerce",
        )

        d.loc[
            ~np.isfinite(d[c]), c
        ] = np.nan

    return d[
        ["mes"] + columnas
    ].reset_index(drop=True)


def leer_tabla(raw, nombre):
    if nombre.lower().endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(raw))

    for codificacion in ("utf-8-sig", "latin-1"):
        for separador in (",", ";"):
            try:
                d = pd.read_csv(
                    io.BytesIO(raw),
                    encoding=codificacion,
                    sep=separador,
                )

                if d.shape[1] >= 2:
                    return d

            except Exception:
                pass

    raise ValueError(
        f"Formato no reconocido: {nombre}"
    )


def calcular_ipc(raw):
    hoja = pd.read_excel(
        io.BytesIO(raw),
        sheet_name="VarCiudades",
        header=None,
    )

    bloques = []

    for fila, valor in enumerate(hoja.iloc[:, 0]):
        m = re.search(
            r"variaciones mensuales\s+(20\d{2})",
            normalizar(valor),
        )

        if m:
            bloques.append(
                (int(m.group(1)), fila)
            )

    if [
        x[0] for x in bloques
    ] != list(range(2017, 2025)):
        raise ValueError(
            "IPC: bloques anuales diferentes del Colab"
        )

    meses = (
        "enero febrero marzo abril mayo junio julio agosto "
        "septiembre octubre noviembre diciembre"
    ).split()

    filas = []

    for k, (anio, inicio) in enumerate(bloques):
        encabezados = [
            normalizar(hoja.iat[inicio + 1, c])
            for c in range(1, 13)
        ]

        if encabezados != meses:
            raise ValueError(
                f"IPC: encabezados incorrectos en {anio}"
            )

        fin = (
            bloques[k + 1][1]
            if k + 1 < len(bloques)
            else len(hoja)
        )

        pasto = [
            j for j in range(inicio + 2, fin)
            if normalizar(hoja.iat[j, 0]) == "pasto"
        ]

        nacional = [
            j for j in range(inicio + 2, fin)
            if normalizar(hoja.iat[j, 0]) == "total ipc"
        ]

        if len(pasto) != 1 or len(nacional) != 1:
            raise ValueError(
                f"IPC: filas Pasto/Nacional ambiguas en {anio}"
            )

        for mes in range(1, 13):
            if anio >= 2018:
                p = numero(
                    hoja.iat[pasto[0], mes]
                )
                n = numero(
                    hoja.iat[nacional[0], mes]
                )

                filas.append({
                    "mes": pd.Period(
                        f"{anio}-{mes:02d}"
                    ),
                    "IPC_MENSUAL_PASTO_PCT": p,
                    "IPC_MENSUAL_NACIONAL_PCT": n,
                    Y: p - n,
                })

    resultado = pd.DataFrame(filas)

    auditar_meses(
        resultado,
        MES_84,
        [
            Y,
            "IPC_MENSUAL_PASTO_PCT",
            "IPC_MENSUAL_NACIONAL_PCT",
        ],
    )

    ultimo = resultado.iloc[-1]

    if not (
        np.isclose(
            ultimo["IPC_MENSUAL_PASTO_PCT"], 0.47
        )
        and np.isclose(
            ultimo["IPC_MENSUAL_NACIONAL_PCT"], 0.46
        )
    ):
        raise ValueError(
            "IPC: control diciembre 2024 no coincide"
        )

    return resultado


def calcular_ictc(raw):
    hoja = pd.read_excel(
        io.BytesIO(raw),
        sheet_name="Anexo1",
        header=None,
    ).iloc[:, :3].copy()

    hoja.columns = [
        "anio_raw",
        "mes_raw",
        "valor_raw",
    ]

    def anio(v):
        try:
            valor = int(float(v))

            return (
                float(valor)
                if 2000 <= valor <= 2100
                else np.nan
            )

        except (ValueError, TypeError):
            return np.nan

    hoja["anio"] = hoja["anio_raw"].map(anio).ffill()

    mapa = {
        m: i + 1
        for i, m in enumerate(
            "enero febrero marzo abril mayo junio julio agosto "
            "septiembre octubre noviembre diciembre".split()
        )
    }

    hoja["n_mes"] = hoja["mes_raw"].map(
        normalizar
    ).map(mapa)

    hoja[I] = hoja["valor_raw"].map(numero)

    hoja = hoja[
        hoja["anio"].between(2018, 2024)
        & hoja["n_mes"].notna()
        & hoja[I].notna()
    ].copy()

    hoja["mes"] = pd.PeriodIndex(
        year=hoja["anio"].astype(int),
        month=hoja["n_mes"].astype(int),
        freq="M",
    )

    resultado = hoja[
        ["mes", I]
    ].sort_values("mes").reset_index(drop=True)

    auditar_meses(
        resultado,
        MES_84,
        [I],
    )

    if not (
        np.isclose(resultado.iloc[0][I], 0.70)
        and np.isclose(resultado.iloc[1][I], 0.83)
    ):
        raise ValueError(
            "ICTC: controles históricos no coinciden"
        )

    return resultado


def descubrir_csv_publicos():
    archivos = []
    errores = []
    vistos = set()

    claves = (
        "sipsa_indicadores",
        "indicadores_mensuales",
        "base_integrada",
        "base_econometrica",
        "base_estimacion",
        "pm_ponderado",
        "variable3_brecha",
        "ictc_mensual",
    )

    for repo in (REPO_SIPSA, REPO_RNDC):
        try:
            metadata = requests.get(
                f"https://api.github.com/repos/{repo}",
                timeout=20,
            )
            metadata.raise_for_status()

            rama = metadata.json().get(
                "default_branch",
                "main",
            )

            ruta = (
                f"https://api.github.com/repos/{repo}/"
                f"git/trees/{rama}?recursive=1"
            )

            respuesta = requests.get(
                ruta,
                timeout=25,
            )
            respuesta.raise_for_status()

            for item in respuesta.json().get("tree", []):
                nombre = item.get("path", "")

                if item.get("type") != "blob":
                    continue

                if not nombre.lower().endswith(
                    (".csv", ".parquet")
                ):
                    continue

                if not any(
                    clave in nombre.lower()
                    for clave in claves
                ):
                    continue

                url = (
                    f"https://raw.githubusercontent.com/"
                    f"{repo}/{rama}/{nombre}"
                )

                if url not in vistos:
                    archivos.append((nombre, url))
                    vistos.add(url)

        except Exception as exc:
            errores.append(
                f"Inventario {repo}: {exc}"
            )

        try:
            releases = requests.get(
                f"https://api.github.com/repos/{repo}/releases",
                timeout=20,
            )
            releases.raise_for_status()

            for release in releases.json()[:10]:
                for asset in release.get("assets", []):
                    nombre = asset["name"]

                    if not nombre.lower().endswith(
                        (".csv", ".parquet")
                    ):
                        continue

                    if not any(
                        clave in nombre.lower()
                        for clave in claves
                    ):
                        continue

                    url = asset["browser_download_url"]

                    if url not in vistos:
                        archivos.append((nombre, url))
                        vistos.add(url)

        except Exception as exc:
            errores.append(
                f"Releases {repo}: {exc}"
            )

    return archivos, errores


@st.cache_data(
    ttl=3600,
    show_spinner="Verificando y recuperando datos oficiales...",
)
def cargar_datos():
    base = pd.DataFrame({
        "mes": MES_84
    })

    partes = []
    bitacora = []
    errores = []

    def trabajo(nombre):
        raw, url = verificar_descarga(nombre)

        parsers = {
            "IPC": calcular_ipc,
            "ICTC": calcular_ictc,
            "RNDC": lambda v: normalizar_base(
                leer_tabla(v, "rndc.csv"),
                "RNDC",
            ),
        }

        tabla = parsers[nombre](raw)

        if nombre == "RNDC":
            auditar_meses(
                tabla,
                MES_59,
                [T],
            )

        return (
            nombre,
            tabla,
            url,
            sha(raw),
            len(raw),
            "SHA-256 verificado",
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        procesos = {
            nombre: executor.submit(trabajo, nombre)
            for nombre in ("IPC", "ICTC", "RNDC")
        }

        for nombre, futuro in procesos.items():
            try:
                partes.append(futuro.result())

            except Exception as exc:
                errores.append(
                    f"{nombre}: {exc}"
                )

    candidatos, incidencias = descubrir_csv_publicos()
    errores.extend(incidencias)

    for nombre, url in candidatos:
        try:
            raw = descargar(url)

            tabla = normalizar_base(
                leer_tabla(raw, nombre),
                nombre,
            )

            partes.append((
                nombre,
                tabla,
                url,
                sha(raw),
                len(raw),
                "SHA calculado, sin referencia histórica",
            ))

        except Exception as exc:
            errores.append(
                f"{nombre}: {exc}"
            )

    archivo = (
        RUNTIME
        / "SIPSA_AUDITORIA_RESULTADOS"
        / "SIPSA_INDICADORES_MENSUALES_2018_2024_CORREGIDO.csv"
    )

    if archivo.is_file():
        try:
            raw = archivo.read_bytes()

            tabla = normalizar_base(
                leer_tabla(raw, archivo.name),
                archivo.name,
            )

            partes.append((
                "SIPSA reconstruido",
                tabla,
                str(archivo),
                sha(raw),
                len(raw),
                "Reconstrucción auditada; SHA calculado",
            ))

        except Exception as exc:
            errores.append(
                f"Reconstrucción SIPSA: {exc}"
            )

    bloqueadas = set()
    procedencia = {}

    for nombre, tabla, url, huella, tam, control in partes:
        bitacora.append({
            "Archivo": nombre,
            "Procedencia": url,
            "SHA256": huella,
            "Tamaño_bytes": tam,
            "Estado": control,
        })

        indice = tabla.set_index("mes")

        for variable in tabla.columns:
            if variable == "mes" or variable in bloqueadas:
                continue

            nuevo = indice[variable].reindex(
                base["mes"]
            ).to_numpy()

            if variable not in base:
                base[variable] = nuevo
                procedencia[variable] = nombre

            else:
                viejo = pd.to_numeric(
                    base[variable],
                    errors="coerce",
                )

                nuevo = pd.Series(
                    nuevo,
                    dtype=float,
                )

                ambos = viejo.notna() & nuevo.notna()

                if ambos.any() and not np.allclose(
                    viejo[ambos],
                    nuevo[ambos],
                    rtol=1e-7,
                    atol=1e-9,
                ):
                    base[variable] = np.nan
                    bloqueadas.add(variable)

                    errores.append(
                        f"Conflicto de datos {variable}: "
                        f"{procedencia[variable]} y {nombre}"
                    )

                else:
                    base[variable] = viejo.combine_first(nuevo)

    return base, pd.DataFrame(bitacora), errores


def reconstruir_sipsa():
    endpoints = [
        (
            "https://drive.google.com/uc?"
            f"export=download&id={NOTEBOOK_ID}"
        ),
        (
            "https://drive.usercontent.google.com/download?"
            f"id={NOTEBOOK_ID}&export=download&confirm=t"
        ),
    ]

    problemas = []

    for url in endpoints:
        try:
            notebook = requests.get(
                url,
                timeout=90,
                headers={"User-Agent": "Mozilla/5.0"},
            )

            notebook.raise_for_status()
            contenido = notebook.json()

            fuente = "".join(
                contenido["cells"][2]["source"]
            )

            if sha(fuente.encode("utf-8")) != SIPSA_CELDA_SHA:
                raise ValueError(
                    "La fase SIPSA pública cambió respecto "
                    "al Colab auditado"
                )

            RUNTIME.mkdir(
                parents=True,
                exist_ok=True,
            )

            rutas = {
                "/content/SIPSA_DANE": str(
                    RUNTIME / "SIPSA_DANE"
                ),
                "/content/SIPSA_GITHUB_CACHE": str(
                    RUNTIME / "SIPSA_GITHUB_CACHE"
                ),
                "/content/SIPSA_AUDITORIA_RESULTADOS": str(
                    RUNTIME / "SIPSA_AUDITORIA_RESULTADOS"
                ),
            }

            for actual, destino in rutas.items():
                fuente = fuente.replace(
                    actual,
                    destino,
                )

            script = (
                RUNTIME / "sipsa_fases_1_7_verificadas.py"
            )

            script.write_text(
                fuente,
                encoding="utf-8",
            )

            resultado = subprocess.run(
                [sys.executable, str(script)],
                cwd=BASE,
                capture_output=True,
                text=True,
                timeout=3600,
            )

            if resultado.returncode:
                raise RuntimeError(
                    (resultado.stderr or resultado.stdout)[-1800:]
                )

            archivo = (
                RUNTIME
                / "SIPSA_AUDITORIA_RESULTADOS"
                / "SIPSA_INDICADORES_MENSUALES_2018_2024_CORREGIDO.csv"
            )

            if not archivo.is_file():
                raise ValueError(
                    "La fase auditada no generó el indicador mensual"
                )

            tabla = normalizar_base(
                pd.read_csv(archivo),
                archivo.name,
            )

            auditar_meses(
                tabla,
                MES_84,
                [P],
            )

            return (
                f"Reconstrucción satisfactoria: 84 meses. "
                f"SHA-256: {sha(archivo.read_bytes())}"
            )

        except Exception as exc:
            problemas.append(str(exc)[:350])

    raise RuntimeError(
        " | ".join(problemas)
    )


def muestra_comun(base, variables, minimo=1):
    faltantes = [
        v for v in variables
        if v not in base
    ]

    if faltantes:
        raise ValueError(
            "Faltan variables: " + ", ".join(faltantes)
        )

    d = base.loc[
        base["mes"].isin(MES_59),
        ["mes"] + variables,
    ].copy()

    if (
        len(d) != 59
        or d[variables].isna().any().any()
    ):
        raise ValueError(
            "No están completos los 59 meses comunes. "
            "No se realizan imputaciones."
        )

    auditar_meses(
        d,
        MES_59,
        variables,
    )

    if len(d) < minimo:
        raise ValueError(
            "Muestra insuficiente"
        )

    return d.reset_index(drop=True)


def construir_modelo(
    base,
    L=2,
    ar=True,
    transporte=T,
    precios=P,
):
    d = muestra_comun(
        base,
        [Y, transporte, precios],
        59,
    )

    z = d[["mes", Y]].copy()
    columnas = []

    if ar:
        z["B_L1"] = d[Y].shift(1)
        columnas.append("B_L1")

    for variable, prefijo in [
        (transporte, "T"),
        (precios, "P"),
    ]:
        for lag in range(L + 1):
            nombre = f"{prefijo}_L{lag}"

            z[nombre] = d[variable].shift(lag)
            columnas.append(nombre)

    z = z[
        z["mes"].isin(MES_56)
    ].dropna().reset_index(drop=True)

    auditar_meses(
        z,
        MES_56,
        [Y] + columnas,
    )

    return z, columnas


def ajustar(
    base,
    L=2,
    ar=True,
    hac=3,
    transporte=T,
    precios=P,
    excluir=None,
    estacionalidad=False,
):
    z, columnas = construir_modelo(
        base,
        L,
        ar,
        transporte,
        precios,
    )

    if excluir is not None:
        z = z[
            z["mes"] != pd.Period(excluir)
        ].copy()

    if estacionalidad:
        trimestre = (
            ((z["mes"].dt.month - 1) // 3) + 1
        )

        for q in (2, 3, 4):
            nombre = f"Q{q}"

            z[nombre] = (
                trimestre == q
            ).astype(int)

            columnas.append(nombre)

    X = sm.add_constant(
        z[columnas].astype(float),
        has_constant="add",
    )

    if len(z) <= X.shape[1] + 8:
        raise ValueError(
            "Grados de libertad insuficientes"
        )

    y = z[Y].astype(float)

    mco = sm.OLS(
        y, X
    ).fit()

    robusto = sm.OLS(
        y, X
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac},
        use_t=True,
    )

    if not np.isfinite(
        robusto.params.to_numpy()
    ).all():
        raise ValueError(
            "Parámetros inválidos"
        )

    return robusto, mco, z, X


def tabla_coef(modelo):
    ic = modelo.conf_int()

    return pd.DataFrame({
        "Regresor": modelo.params.index,
        "Coeficiente": modelo.params.values,
        "Error_HAC": modelo.bse.values,
        "p_HAC": modelo.pvalues.values,
        "IC_inf": ic.iloc[:, 0].values,
        "IC_sup": ic.iloc[:, 1].values,
    })


def contraste_bloques(modelo, L=2):
    filas = []

    for nombre, prefijo in [
        ("Transporte", "T"),
        ("SIPSA", "P"),
    ]:
        columnas = [
            f"{prefijo}_L{j}"
            for j in range(L + 1)
        ]

        R = np.zeros(
            (L + 1, len(modelo.params))
        )

        for i, col in enumerate(columnas):
            R[
                i,
                modelo.params.index.get_loc(col)
            ] = 1

        prueba_f = modelo.wald_test(
            R,
            use_f=True,
            scalar=True,
        )

        prueba_suma = modelo.t_test(
            R.sum(axis=0)
        )

        filas.append({
            "Bloque": nombre,
            "p_conjunto_HAC": float(prueba_f.pvalue),
            "Suma": float(
                modelo.params[columnas].sum()
            ),
            "IC_suma_inf": float(
                prueba_suma.conf_int()[0, 0]
            ),
            "IC_suma_sup": float(
                prueba_suma.conf_int()[0, 1]
            ),
        })

    return pd.DataFrame(filas)


def diagnosticar(mco, z, X):
    e = mco.resid

    valores = {
        "Jarque_Bera_p": float(
            jarque_bera(e)[1]
        ),
        "LjungBox_6_p": float(
            acorr_ljungbox(
                e,
                lags=[6],
                return_df=True,
            )["lb_pvalue"].iloc[0]
        ),
        "LjungBox_12_p": float(
            acorr_ljungbox(
                e,
                lags=[12],
                return_df=True,
            )["lb_pvalue"].iloc[0]
        ),
        "BreuschGodfrey_3_p": float(
            acorr_breusch_godfrey(
                mco,
                nlags=3,
            )[1]
        ),
        "BreuschPagan_p": float(
            het_breuschpagan(e, X)[1]
        ),
        "CUSUM_p": float(
            breaks_cusumolsresid(
                e,
                ddof=X.shape[1],
            )[1]
        ),
    }

    try:
        valores["RESET_p"] = float(
            linear_reset(
                mco,
                power=2,
                use_f=True,
            ).pvalue
        )

    except Exception:
        valores["RESET_p"] = np.nan

    valores["Numero_condicion"] = float(
        np.linalg.cond(X)
    )

    vif = []

    for i, col in enumerate(X.columns):
        if col != "const":
            vif.append({
                "Regresor": col,
                "VIF": variance_inflation_factor(
                    X.values, i
                ),
            })

    datos = z[
        ["mes", Y]
    ].copy()

    datos["Ajuste"] = (
        mco.fittedvalues.to_numpy()
    )

    datos["Residuo"] = e.to_numpy()

    datos["Cook"] = (
        OLSInfluence(mco).cooks_distance[0]
    )

    return (
        pd.DataFrame([valores]),
        datos,
        pd.DataFrame(vif),
    )


def tabla_robustez(base):
    opciones = [
        (
            f"{familia} L{L}",
            dict(
                L=L,
                ar=(familia == "ARDL"),
            ),
        )
        for familia in ("DL", "ARDL")
        for L in range(4)
    ]

    opciones += [
        (f"HAC({h})", dict(hac=h))
        for h in (1, 6)
    ]

    opciones += [
        (
            "Excluir 2023-02",
            dict(excluir="2023-02"),
        ),
        (
            "Dummies trimestrales",
            dict(estacionalidad=True),
        ),
    ]

    alternativas_precios = (
        "PM_no_ponderado_t",
        "PM_ponderado_lagQ_t",
        "PM_100",
        "PM_95",
    )

    for alternativa in alternativas_precios:
        if (
            alternativa in base
            and base[alternativa].notna().any()
        ):
            opciones.append((
                f"SIPSA {alternativa}",
                dict(precios=alternativa),
            ))

    alternativas_transporte = (
        I,
        "VAR_FLETE_FISHER_PCT",
        "VAR_FLETE_CRUDO_PCT",
        "VAR_FLETE_SIN_2000_PCT",
    )

    for alternativa in alternativas_transporte:
        if (
            alternativa in base
            and base[alternativa].notna().any()
        ):
            opciones.append((
                f"Transporte {alternativa}",
                dict(transporte=alternativa),
            ))

    filas = []

    for nombre, kwargs in opciones:
        try:
            m, o, z, _ = ajustar(
                base,
                **kwargs,
            )

            prueba = contraste_bloques(
                m,
                kwargs.get("L", 2),
            )

            filas.append({
                "Especificación": nombre,
                "N": len(z),
                "R2_aj": o.rsquared_adj,
                "AIC": o.aic,
                "BIC": o.bic,
                "p_transporte": prueba.iloc[0]["p_conjunto_HAC"],
                "p_SIPSA": prueba.iloc[1]["p_conjunto_HAC"],
                "Suma_transporte": prueba.iloc[0]["Suma"],
                "Estado": "OK",
            })

        except Exception as exc:
            filas.append({
                "Especificación": nombre,
                "Estado": str(exc)[:140],
            })

    return pd.DataFrame(filas)


def ventanas_moviles(base, ancho=36):
    z, columnas = construir_modelo(base)

    filas = []

    for inicio in range(len(z) - ancho + 1):
        v = z.iloc[
            inicio:inicio + ancho
        ]

        X = sm.add_constant(
            v[columnas],
            has_constant="add",
        )

        m = sm.OLS(
            v[Y], X
        ).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 3},
            use_t=True,
        )

        pruebas = contraste_bloques(m)

        filas.append({
            "Inicio": str(v["mes"].iloc[0]),
            "Fin": str(v["mes"].iloc[-1]),
            "RNDC_suma": pruebas.iloc[0]["Suma"],
            "RNDC_p": pruebas.iloc[0]["p_conjunto_HAC"],
            "SIPSA_suma": pruebas.iloc[1]["Suma"],
        })

    return pd.DataFrame(filas)


def comparar_huber(base):
    z, columnas = construir_modelo(base)

    X = sm.add_constant(
        z[columnas],
        has_constant="add",
    )

    mco = sm.OLS(
        z[Y], X
    ).fit()

    rlm = sm.RLM(
        z[Y],
        X,
        M=sm.robust.norms.HuberT(),
    ).fit()

    filas = []

    for nombre, prefijo in [
        ("RNDC", "T"),
        ("SIPSA", "P"),
    ]:
        columnas_bloque = [
            f"{prefijo}_L{j}"
            for j in range(3)
        ]

        filas.append({
            "Bloque": nombre,
            "Suma_MCO": float(
                mco.params[columnas_bloque].sum()
            ),
            "Suma_Huber": float(
                rlm.params[columnas_bloque].sum()
            ),
        })

    return pd.DataFrame(filas)


def pruebas_estacionariedad(base, variables):
    filas = []

    for col in variables:
        try:
            d = muestra_comun(
                base,
                [col],
                24,
            )

            a = adfuller(
                d[col],
                autolag="AIC",
            )

            k = kpss(
                d[col],
                regression="c",
                nlags="auto",
            )

            filas.append({
                "Variable": NOMBRES.get(col, col),
                "N": len(d),
                "ADF": a[0],
                "p_ADF": a[1],
                "KPSS": k[0],
                "p_KPSS_acotado": k[1],
                "I0_orientativo": (
                    a[1] < 0.05
                    and k[1] >= 0.05
                ),
            })

        except Exception as exc:
            filas.append({
                "Variable": NOMBRES.get(col, col),
                "Estado": str(exc)[:120],
            })

    return pd.DataFrame(filas)


def pca_kmeans(base, variables, k):
    d = muestra_comun(
        base,
        variables,
        59,
    )

    X = StandardScaler().fit_transform(
        d[variables]
    )

    pca = PCA(n_components=2)

    plano = pca.fit_transform(X)

    grupos = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    ).fit_predict(X)

    resultado = pd.DataFrame({
        "mes": d["mes"].astype(str),
        "PC1": plano[:, 0],
        "PC2": plano[:, 1],
        "Grupo": grupos.astype(str),
    })

    cargas = pd.DataFrame(
        pca.components_.T,
        index=variables,
        columns=["PC1", "PC2"],
    )

    return (
        resultado,
        cargas,
        pca.explained_variance_ratio_,
    )


def var_condicional(base, variables, lags, horizonte):
    d = muestra_comun(
        base,
        variables,
        59,
    )

    pruebas = pruebas_estacionariedad(
        base,
        variables,
    )

    if not pruebas["I0_orientativo"].fillna(False).all():
        raise ValueError(
            "VAR en niveles no habilitado: "
            "no hay evidencia conjunta de I(0)"
        )

    modelo = VAR(
        d[variables]
    ).fit(lags)

    if not modelo.is_stable():
        raise ValueError(
            "VAR inestable: se omiten las IRF"
        )

    return (
        modelo,
        modelo.irf(horizonte).irfs,
    )


def evaluar_vecm(base, variables):
    d = muestra_comun(
        base,
        variables,
        59,
    )

    filas = []

    for variable in variables:
        filas.append({
            "Variable": variable,
            "ADF_p_nivel": adfuller(
                d[variable],
                autolag="AIC",
            )[1],
            "ADF_p_diferencia": adfuller(
                d[variable].diff().dropna(),
                autolag="AIC",
            )[1],
        })

    tabla = pd.DataFrame(filas)

    condicion = (
        (tabla["ADF_p_nivel"] >= 0.05)
        & (tabla["ADF_p_diferencia"] < 0.05)
    )

    if not condicion.all():
        return tabla, None

    johansen = coint_johansen(
        d[variables],
        det_order=0,
        k_ar_diff=1,
    )

    resultados = pd.DataFrame({
        "Traza": johansen.lr1,
        "Crítico_95": johansen.cvt[:, 1],
    })

    return tabla, resultados


def multiplicadores(modelo, variable, cambio, horizonte):
    parametros = list(
        modelo.params.index
    )

    cov = modelo.cov_params().to_numpy(float)

    rho = float(
        modelo.params.get("B_L1", 0.0)
    )

    posicion_rho = (
        parametros.index("B_L1")
        if "B_L1" in parametros
        else None
    )

    resultado = 0.0
    derivada = np.zeros(len(parametros))
    filas = []

    for h in range(horizonte + 1):
        previo = resultado
        previo_g = derivada.copy()

        prefijo = "T" if variable == T else "P"
        nombre = f"{prefijo}_L{h}"

        resultado = (
            rho * previo
            + float(modelo.params.get(nombre, 0.0))
        )

        derivada = rho * previo_g

        if posicion_rho is not None:
            derivada[posicion_rho] += previo

        if nombre in parametros:
            derivada[parametros.index(nombre)] += 1

        varianza = float(
            derivada @ cov @ derivada
        )

        error = np.sqrt(max(varianza, 0.0))

        filas.append({
            "Horizonte": h,
            "Respuesta_pp": cambio * resultado,
            "IC95_inf": (
                cambio * resultado
                - 1.96 * abs(cambio) * error
            ),
            "IC95_sup": (
                cambio * resultado
                + 1.96 * abs(cambio) * error
            ),
        })

    return pd.DataFrame(filas)


def evaluar_ml(base, elegidos):
    d = muestra_comun(
        base,
        [Y, T, P],
        59,
    )

    tabla = pd.DataFrame({
        "mes": d["mes"],
        "objetivo": d[Y],
    })

    for variable in (Y, T, P):
        for lag in (1, 2):
            tabla[f"{variable}_L{lag}"] = (
                d[variable].shift(lag)
            )

    tabla = tabla.dropna().reset_index(drop=True)

    variables = [
        c for c in tabla.columns
        if c not in ("mes", "objetivo")
    ]

    inicio = len(tabla) - 26

    if inicio < 20:
        raise ValueError(
            "Muestra de entrenamiento insuficiente"
        )

    filas = []

    for t in range(inicio, len(tabla)):
        train = tabla.iloc[:t]
        test = tabla.iloc[[t]]

        X = train[variables].to_numpy(float)
        Xt = test[variables].to_numpy(float)
        y = train["objetivo"].to_numpy(float)

        escalador = StandardScaler().fit(X)

        valores = {
            "Media histórica": float(y.mean())
        }

        ar_x = sm.add_constant(
            train[[f"{Y}_L1"]],
            has_constant="add",
        )

        ar_xt = sm.add_constant(
            test[[f"{Y}_L1"]],
            has_constant="add",
        )

        ajuste_ar = sm.OLS(
            y, ar_x
        ).fit()

        valores["AR(1)"] = float(
            ajuste_ar.predict(ar_xt).iloc[0]
        )

        ajuste_arx = sm.OLS(
            y,
            sm.add_constant(
                X,
                has_constant="add",
            ),
        ).fit()

        valores["ARX"] = float(
            ajuste_arx.predict(
                sm.add_constant(
                    Xt,
                    has_constant="add",
                )
            )[0]
        )

        modelos = {
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(
                alpha=0.01,
                max_iter=15000,
            ),
            "SVR": SVR(
                C=1.0,
                epsilon=0.1,
            ),
            "MLP": MLPRegressor(
                hidden_layer_sizes=(12,),
                max_iter=400,
                random_state=42,
            ),
        }

        for nombre, modelo in modelos.items():
            if nombre in elegidos:
                modelo.fit(
                    escalador.transform(X),
                    y,
                )

                valores[nombre] = float(
                    modelo.predict(
                        escalador.transform(Xt)
                    )[0]
                )

        for nombre, prediccion in valores.items():
            if nombre in elegidos:
                filas.append({
                    "mes": str(test["mes"].iloc[0]),
                    "Modelo": nombre,
                    "Observado": float(
                        test["objetivo"].iloc[0]
                    ),
                    "Predicho": prediccion,
                })

    pronosticos = pd.DataFrame(filas)

    resumen = pd.DataFrame([
        {
            "Modelo": nombre,
            "RMSE": np.sqrt(
                mean_squared_error(
                    grupo["Observado"],
                    grupo["Predicho"],
                )
            ),
            "MAE": mean_absolute_error(
                grupo["Observado"],
                grupo["Predicho"],
            ),
        }
        for nombre, grupo in pronosticos.groupby("Modelo")
    ]).sort_values("RMSE")

    return (
        resumen,
        pronosticos,
        tabla,
        variables,
        inicio,
    )


def xai(datos, variables, inicio, modelo_nombre):
    modelos = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(
            alpha=0.01,
            max_iter=15000,
        ),
        "SVR": SVR(
            C=1.0,
            epsilon=0.1,
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(12,),
            max_iter=400,
            random_state=42,
        ),
    }

    X = datos[variables].to_numpy(float)
    y = datos["objetivo"].to_numpy(float)

    escalador = StandardScaler().fit(
        X[:inicio]
    )

    modelo = modelos[modelo_nombre].fit(
        escalador.transform(X[:inicio]),
        y[:inicio],
    )

    importancia = permutation_importance(
        modelo,
        escalador.transform(X[inicio:]),
        y[inicio:],
        n_repeats=10,
        random_state=42,
        scoring="neg_mean_squared_error",
    )

    return pd.DataFrame({
        "Regresor": variables,
        "Importancia": importancia.importances_mean,
        "Desviación": importancia.importances_std,
    }).sort_values("Importancia")


def senales_historicas(base):
    d = muestra_comun(
        base,
        [T, P],
        59,
    )

    filas = []

    for i in range(12, len(d)):
        anteriores = d.iloc[:i]
        actual = d.iloc[i]

        umbral_t = anteriores[T].quantile(0.90)
        umbral_p = anteriores[P].quantile(0.90)

        alto_t = actual[T] > umbral_t
        alto_p = actual[P] > umbral_p

        if alto_t and alto_p:
            senal = "Rojo"
        elif alto_t or alto_p:
            senal = "Amarillo"
        else:
            senal = "Verde"

        filas.append({
            "mes": str(actual["mes"]),
            "RNDC": actual[T],
            "SIPSA": actual[P],
            "Umbral_RNDC": umbral_t,
            "Umbral_SIPSA": umbral_p,
            "Señal": senal,
        })

    return pd.DataFrame(filas)


def figura(fig, altura=430):
    fig.update_layout(
        **STYLE,
        height=altura,
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=35,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    return fig


def exportar(df, nombre, etiqueta="Descargar CSV"):
    d = df.copy()

    if "mes" in d:
        d["mes"] = d["mes"].astype(str)

    st.download_button(
        etiqueta,
        d.to_csv(index=False).encode("utf-8-sig"),
        file_name=nombre,
        mime="text/csv",
    )


def grafico_series(base, variables, titulo):
    datos = base[
        ["mes"] + variables
    ].copy()

    datos["mes"] = datos["mes"].astype(str)

    largo = datos.melt(
        id_vars="mes",
        var_name="Variable",
        value_name="Valor",
    )

    largo["Variable"] = largo["Variable"].map(
        lambda c: NOMBRES.get(c, c)
    )

    if len(variables) > 1:
        graf = px.line(
            largo,
            x="mes",
            y="Valor",
            color="Variable",
            facet_row="Variable",
            title=titulo,
        )

        graf.for_each_yaxis(
            lambda eje: eje.update(matches=None)
        )

        return figura(
            graf,
            altura=max(430, len(variables) * 210),
        )

    return figura(
        px.line(
            largo,
            x="mes",
            y="Valor",
            color="Variable",
            title=titulo,
        )
    )


def disponibles(base, variables):
    faltantes = [
        c for c in variables
        if c not in base
        or base[c].notna().sum() == 0
    ]

    if faltantes:
        st.warning(
            "No hay datos suficientes: "
            + ", ".join(
                NOMBRES.get(x, x)
                for x in faltantes
            )
        )

    return not faltantes


with st.sidebar:
    st.title("📊 Observatorio Pasto")

    pagina = st.radio(
        "Navegación",
        [
            "🏠 Inicio",
            "🗂️ Datos y muestras",
            "📊 Análisis Pasto–Nariño",
            "📑 Generador de informes",
            "💬 Chatbot",
            "© Fuentes y créditos",
        ],
    )

    st.divider()

    inicio, fin = st.select_slider(
        "Periodo descriptivo",
        options=[
            str(x) for x in MES_84
        ],
        value=("2018-01", "2024-12"),
    )

    st.caption(
        "Los modelos del artículo conservan la muestra "
        "de 56 meses y no cambian con este filtro."
    )

    if st.button("Actualizar fuentes"):
        cargar_datos.clear()
        st.rerun()


try:
    DATA, AUDITORIA, ERRORES = cargar_datos()

except Exception as exc:
    st.error(
        f"No se pudo construir el observatorio: {exc}"
    )
    st.stop()


VARS = [
    c for c in (Y, T, P, I)
    if c in DATA
    and DATA[c].notna().any()
]

VISTA = DATA[
    (DATA["mes"] >= pd.Period(inicio))
    & (DATA["mes"] <= pd.Period(fin))
].copy()


if pagina == "🏠 Inicio":
    st.title(
        "Observatorio económico de Pasto y Nariño"
    )

    st.write(
        "Costos del transporte de carga, precios mayoristas "
        "de El Potrerillo y brecha inflacionaria "
        "Pasto–Colombia · 2018–2024"
    )

    st.info(
        "Pasto es la ciudad analizada. Los resultados "
        "no representan automáticamente todo Nariño."
    )

    columnas = st.columns(4)

    for panel, variable in zip(
        columnas,
        [Y, P, T, I],
    ):
        n = (
            int(DATA[variable].notna().sum())
            if variable in DATA
            else 0
        )

        panel.metric(
            NOMBRES[variable],
            f"{n} meses" if n else "No disponible",
        )

    if VARS:
        grafico_series(
            VISTA,
            VARS,
            "Evolución mensual de las series observadas",
        )

    st.markdown(
        "**Estructura:** 84 meses descriptivos; "
        "59 meses comunes; 56 observaciones para DL/ARDL. "
        "Fuentes: DANE y RNDC del Ministerio de Transporte, "
        "con respaldos en GitHub."
    )

    st.warning(
        "Análisis estadístico asociativo. "
        "No constituye un sistema validado de "
        "predicción de cierres viales."
    )


elif pagina == "🗂️ Datos y muestras":
    st.header("Datos, cobertura y auditoría")

    cobertura = []

    for variable in [Y, T, P, I]:
        fechas = (
            DATA.loc[
                DATA[variable].notna(),
                "mes",
            ]
            if variable in DATA
            else pd.Series(dtype=object)
        )

        cobertura.append({
            "Indicador": NOMBRES[variable],
            "N": len(fechas),
            "Desde": (
                str(fechas.min())
                if len(fechas)
                else "—"
            ),
            "Hasta": (
                str(fechas.max())
                if len(fechas)
                else "—"
            ),
        })

    st.dataframe(
        pd.DataFrame(cobertura),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Fuentes y SHA-256")

    st.dataframe(
        AUDITORIA,
        hide_index=True,
        use_container_width=True,
    )

    exportar(
        AUDITORIA,
        "auditoria_fuentes.csv",
    )

    with st.expander(
        f"Incidencias ({len(ERRORES)})"
    ):
        for error in ERRORES:
            st.write(error)

    st.dataframe(
        DATA.assign(
            mes=DATA["mes"].astype(str)
        ),
        use_container_width=True,
    )

    exportar(
        DATA,
        "base_integrada_observada.csv",
    )

    if (
        P not in DATA
        or DATA[P].notna().sum() != 84
    ):
        st.warning(
            "SIPSA no está completo. La reconstrucción "
            "utiliza las fases 1–7 del Colab verificadas "
            "por SHA-256 y los ZIP del DANE respaldados "
            "en GitHub."
        )

        if st.button(
            "Reconstruir SIPSA sin subir archivos",
            type="primary",
        ):
            try:
                st.success(reconstruir_sipsa())
                cargar_datos.clear()
                st.rerun()

            except Exception as exc:
                st.error(
                    f"Reconstrucción no completada: {exc}"
                )


elif pagina == "📊 Análisis Pasto–Nariño":
    st.header(
        "Herramientas estadísticas y econométricas"
    )

    tabs = st.tabs([
        "EDA",
        "ADF / KPSS",
        "ARDL + HAC",
        "Diagnósticos",
        "Robustez",
        "PCA / K-Means",
        "VAR / VECM",
        "ML / XAI",
        "Simulador",
        "Focus Pasto",
    ])

    with tabs[0]:
        elegir = st.multiselect(
            "Series para EDA",
            VARS,
            default=VARS,
            format_func=lambda c: NOMBRES[c],
        )

        if elegir:
            st.dataframe(
                VISTA[elegir].describe().T,
                use_container_width=True,
            )

            grafico_series(
                VISTA,
                elegir,
                "Series seleccionadas",
            )

            if len(elegir) > 1:
                correlacion = VISTA[elegir].corr(
                    min_periods=12
                )

                figura(
                    px.imshow(
                        correlacion,
                        text_auto=".3f",
                        zmin=-1,
                        zmax=1,
                        color_continuous_scale="RdBu_r",
                        title="Correlación de Pearson",
                    )
                )

            exportar(
                VISTA[["mes"] + elegir],
                "series_eda.csv",
            )

    with tabs[1]:
        if VARS:
            estacionarias = pruebas_estacionariedad(
                DATA,
                VARS,
            )

            st.dataframe(
                estacionarias,
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "ADF: nula de raíz unitaria. "
                "KPSS: nula de estacionariedad. "
                "Los valores p de KPSS son tabulados."
            )

            exportar(
                estacionarias,
                "estacionariedad.csv",
            )

    with tabs[2]:
        st.latex(
            r"B_t=\alpha+\rho B_{t-1}"
            r"+\sum_{j=0}^{L}\beta_jT_{t-j}"
            r"+\sum_{j=0}^{L}\gamma_jPM_{t-j}+u_t"
        )

        c1, c2, c3 = st.columns(3)

        L = c1.slider(
            "Rezagos", 0, 3, 2
        )

        ar = c2.checkbox(
            "Incluir brecha rezagada",
            True,
        )

        hac = c3.slider(
            "Ancho HAC", 0, 8, 3
        )

        if disponibles(DATA, [Y, T, P]):
            try:
                modelo, mco, z, X = ajustar(
                    DATA,
                    L=L,
                    ar=ar,
                    hac=hac,
                )

                c1, c2, c3 = st.columns(3)

                c1.metric("N", len(z))

                c2.metric(
                    "R² ajustado",
                    f"{mco.rsquared_adj:.4f}",
                )

                c3.metric(
                    "AIC",
                    f"{mco.aic:.4f}",
                )

                coef = tabla_coef(modelo)

                bloques = contraste_bloques(
                    modelo,
                    L,
                )

                st.dataframe(
                    coef,
                    hide_index=True,
                    use_container_width=True,
                )

                st.dataframe(
                    bloques,
                    hide_index=True,
                    use_container_width=True,
                )

                exportar(
                    coef,
                    "coeficientes_hac.csv",
                )

                exportar(
                    bloques,
                    "bloques_hac.csv",
                )

                ajuste = z[
                    ["mes", Y]
                ].copy()

                ajuste["Ajustado"] = (
                    mco.fittedvalues.to_numpy()
                )

                grafico_series(
                    ajuste,
                    [Y, "Ajustado"],
                    "Brecha observada y ajuste dentro de muestra",
                )

                if (L, ar, hac) == (2, True, 3):
                    referencia = {
                        "Coef_RNDC_t": 0.00913,
                        "p_RNDC_t": 0.03750,
                        "p_conjunto_RNDC": 0.14273,
                        "p_conjunto_SIPSA": 0.23749,
                    }

                    calculados = {
                        "Coef_RNDC_t": modelo.params["T_L0"],
                        "p_RNDC_t": modelo.pvalues["T_L0"],
                        "p_conjunto_RNDC": (
                            bloques.iloc[0]["p_conjunto_HAC"]
                        ),
                        "p_conjunto_SIPSA": (
                            bloques.iloc[1]["p_conjunto_HAC"]
                        ),
                    }

                    st.caption(
                        "Contraste con valores redondeados del artículo"
                    )

                    comparacion = pd.DataFrame([
                        {
                            "Control": clave,
                            "Artículo": valor,
                            "App": calculados[clave],
                            "Diferencia": abs(
                                calculados[clave] - valor
                            ),
                        }
                        for clave, valor in referencia.items()
                    ])

                    st.dataframe(
                        comparacion,
                        hide_index=True,
                    )

                st.info(
                    "La significancia individual no establece "
                    "significancia de los bloques ni causalidad."
                )

            except Exception as exc:
                st.error(str(exc))

    with tabs[3]:
        if disponibles(DATA, [Y, T, P]):
            try:
                _, mco, z, X = ajustar(DATA)

                diag, residuos, vif = diagnosticar(
                    mco,
                    z,
                    X,
                )

                st.dataframe(diag.T)

                st.dataframe(
                    vif,
                    hide_index=True,
                )

                figura(
                    px.scatter(
                        residuos,
                        x="Ajuste",
                        y="Residuo",
                        title="Residuos frente al ajuste",
                    )
                )

                figura(
                    px.bar(
                        residuos.assign(
                            mes=residuos["mes"].astype(str)
                        ),
                        x="mes",
                        y="Cook",
                        title="Distancias de Cook",
                    )
                )

                exportar(
                    residuos,
                    "residuos_ardl.csv",
                )

            except Exception as exc:
                st.error(str(exc))

    with tabs[4]:
        if disponibles(DATA, [Y, T, P]):
            if st.button(
                "Comparar 8 modelos DL/ARDL y sensibilidades"
            ):
                resultado = tabla_robustez(DATA)

                st.dataframe(
                    resultado,
                    hide_index=True,
                    use_container_width=True,
                )

                exportar(
                    resultado,
                    "robustez.csv",
                )

            if st.button(
                "Comparar MCO frente a Huber"
            ):
                try:
                    st.dataframe(
                        comparar_huber(DATA),
                        hide_index=True,
                    )

                except Exception as exc:
                    st.error(str(exc))

            if st.button(
                "Ventanas móviles de 36 meses"
            ):
                try:
                    ventanas = ventanas_moviles(DATA)

                    st.dataframe(
                        ventanas,
                        hide_index=True,
                    )

                    exportar(
                        ventanas,
                        "ventanas.csv",
                    )

                except Exception as exc:
                    st.error(str(exc))

    with tabs[5]:
        seleccion = st.multiselect(
            "Series para PCA",
            VARS,
            default=VARS,
            format_func=lambda x: NOMBRES[x],
        )

        clusters = st.slider(
            "K clusters",
            2,
            6,
            3,
        )

        if (
            len(seleccion) >= 2
            and st.button("Ejecutar PCA y K-Means")
        ):
            try:
                coords, cargas, proporcion = pca_kmeans(
                    DATA,
                    seleccion,
                    clusters,
                )

                st.write(
                    f"Varianza explicada: "
                    f"PC1 {proporcion[0]:.1%}; "
                    f"PC2 {proporcion[1]:.1%}"
                )

                figura(
                    px.scatter(
                        coords,
                        x="PC1",
                        y="PC2",
                        color="Grupo",
                        hover_data=["mes"],
                        title="PCA y clusters de meses",
                    )
                )

                st.dataframe(cargas)

                exportar(
                    coords,
                    "clusters_mensuales.csv",
                )

                st.caption(
                    "Se agrupan meses, no departamentos. "
                    "No se realizan imputaciones."
                )

            except Exception as exc:
                st.error(str(exc))

    with tabs[6]:
        elegir = st.multiselect(
            "Variables endógenas",
            VARS,
            default=[
                x for x in (Y, T, P)
                if x in VARS
            ],
            max_selections=3,
            format_func=lambda x: NOMBRES[x],
        )

        lags = st.slider(
            "Rezagos VAR",
            1,
            3,
            1,
        )

        if (
            len(elegir) >= 2
            and st.button("Estimar VAR si es admisible")
        ):
            try:
                m, irf = var_condicional(
                    DATA,
                    elegir,
                    lags,
                    6,
                )

                st.write({
                    "AIC": m.aic,
                    "BIC": m.bic,
                    "Estable": m.is_stable(),
                })

                impacto = st.selectbox(
                    "Impulso",
                    elegir,
                )

                respuesta = st.selectbox(
                    "Respuesta",
                    elegir,
                )

                valores = irf[
                    :,
                    elegir.index(respuesta),
                    elegir.index(impacto),
                ]

                figura(
                    px.line(
                        x=list(range(len(valores))),
                        y=valores,
                        markers=True,
                        title="IRF reducida exploratoria",
                    )
                )

            except Exception as exc:
                st.info(str(exc))

        if (
            len(elegir) >= 2
            and st.button("Revisar elegibilidad VECM/Johansen")
        ):
            try:
                tabla, johansen = evaluar_vecm(
                    DATA,
                    elegir,
                )

                st.dataframe(
                    tabla,
                    hide_index=True,
                )

                if johansen is not None:
                    st.dataframe(
                        johansen,
                        hide_index=True,
                    )

                else:
                    st.info(
                        "No se justifica VECM con la evidencia "
                        "de integración disponible."
                    )

            except Exception as exc:
                st.warning(str(exc))

    with tabs[7]:
        if disponibles(DATA, [Y, T, P]):
            elegidos = st.multiselect(
                "Modelos predictivos",
                [
                    "Media histórica",
                    "AR(1)",
                    "ARX",
                    "Ridge",
                    "Lasso",
                    "SVR",
                    "MLP",
                ],
                default=[
                    "Media histórica",
                    "AR(1)",
                    "ARX",
                    "Ridge",
                    "Lasso",
                    "SVR",
                ],
            )

            if (
                elegidos
                and st.button("Validación cronológica y XAI")
            ):
                try:
                    resumen, pronosticos, datos, features, inicio = evaluar_ml(
                        DATA,
                        elegidos,
                    )

                    st.dataframe(
                        resumen,
                        hide_index=True,
                        use_container_width=True,
                    )

                    figura(
                        px.line(
                            pronosticos,
                            x="mes",
                            y="Predicho",
                            color="Modelo",
                            title="Pronósticos exploratorios",
                        )
                    )

                    exportar(
                        pronosticos,
                        "pronosticos_ml.csv",
                    )

                    xai_models = [
                        x for x in elegidos
                        if x in (
                            "Ridge",
                            "Lasso",
                            "SVR",
                            "MLP",
                        )
                    ]

                    if xai_models:
                        importancia = xai(
                            datos,
                            features,
                            inicio,
                            xai_models[0],
                        )

                        figura(
                            px.bar(
                                importancia,
                                x="Importancia",
                                y="Regresor",
                                orientation="h",
                                title=(
                                    "XAI por permutación: "
                                    f"{xai_models[0]}"
                                ),
                            )
                        )

                        exportar(
                            importancia,
                            "importancia_xai.csv",
                        )

                    st.warning(
                        "Validación seudofuera de muestra: "
                        "el calendario histórico de publicación "
                        "no está comprobado. No detecta bloqueos."
                    )

                except Exception as exc:
                    st.error(str(exc))

    with tabs[8]:
        if disponibles(DATA, [Y, T, P]):
            variable = st.selectbox(
                "Variable del escenario",
                [T, P],
                format_func=lambda x: NOMBRES[x],
            )

            cambio = st.number_input(
                "Cambio hipotético del indicador (p. p.)",
                value=1.0,
                step=0.5,
            )

            horizonte = st.slider(
                "Meses de respuesta",
                1,
                12,
                6,
            )

            if st.button(
                "Simular trayectoria condicionada"
            ):
                try:
                    modelo, _, _, _ = ajustar(DATA)

                    resultado = multiplicadores(
                        modelo,
                        variable,
                        cambio,
                        horizonte,
                    )

                    graf = go.Figure()

                    graf.add_trace(
                        go.Scatter(
                            x=resultado["Horizonte"],
                            y=resultado["IC95_sup"],
                            line=dict(width=0),
                            showlegend=False,
                        )
                    )

                    graf.add_trace(
                        go.Scatter(
                            x=resultado["Horizonte"],
                            y=resultado["IC95_inf"],
                            fill="tonexty",
                            name="IC 95 %",
                        )
                    )

                    graf.add_trace(
                        go.Scatter(
                            x=resultado["Horizonte"],
                            y=resultado["Respuesta_pp"],
                            mode="lines+markers",
                            name="Respuesta",
                        )
                    )

                    figura(graf)

                    exportar(
                        resultado,
                        "simulacion_ardl.csv",
                    )

                    st.caption(
                        "Escenario condicional. No es un choque "
                        "causalmente identificado."
                    )

                except Exception as exc:
                    st.error(str(exc))

    with tabs[9]:
        st.write(
            "En la primera quincena de mayo de 2021, "
            "el DANE documentó una caída del abastecimiento "
            "de El Potrerillo de 4.618 a 1.987 toneladas "
            "(-56,98 %). La evidencia es quincenal; "
            "las series del observatorio son mensuales."
        )

        episodio = DATA[
            DATA["mes"].between(
                pd.Period("2021-04"),
                pd.Period("2021-08"),
            )
        ]

        columnas = [
            c for c in (Y, T, P)
            if c in VARS
        ]

        if columnas:
            st.dataframe(
                episodio[
                    ["mes"] + columnas
                ].assign(
                    mes=lambda v: v["mes"].astype(str)
                ),
                hide_index=True,
            )

            grafico_series(
                episodio,
                columnas,
                "Abril–agosto de 2021",
            )

        if disponibles(DATA, [T, P]):
            try:
                senales = senales_historicas(DATA)

                st.dataframe(
                    senales[
                        senales["mes"].str.startswith("2021")
                    ],
                    hide_index=True,
                )

                exportar(
                    senales,
                    "semaforo_retrospectivo.csv",
                )

                st.warning(
                    "Semáforo retrospectivo por percentiles. "
                    "No es una alerta vial operativa."
                )

            except Exception as exc:
                st.warning(str(exc))


elif pagina == "📑 Generador de informes":
    st.header(
        "Informe automatizado de la investigación"
    )

    contenido = [
        "# Informe estadístico Pasto–Colombia",
        f"Generado: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## Fuentes y cobertura",
    ]

    for variable in (Y, T, P, I):
        n = (
            int(DATA[variable].notna().sum())
            if variable in DATA
            else 0
        )

        contenido.append(
            f"- {NOMBRES[variable]}: {n} meses"
        )

    contenido += [
        "",
        "## Diseño",
        (
            "Estudio asociativo con series mensuales "
            "2018–2024 y muestra econométrica de "
            "56 meses, sin imputaciones."
        ),
    ]

    if all(
        variable in VARS
        for variable in (Y, T, P)
    ):
        try:
            m, o, z, _ = ajustar(DATA)

            bloques = contraste_bloques(m)

            contenido += [
                "",
                "## ARDL L2 / HAC(3)",
                (
                    f"N = {len(z)}; "
                    f"R² ajustado = {o.rsquared_adj:.5f}; "
                    f"AIC = {o.aic:.5f}."
                ),
                (
                    f"Coeficiente RNDC(t) = "
                    f"{m.params['T_L0']:.6f}; "
                    f"p HAC = {m.pvalues['T_L0']:.5f}."
                ),
                (
                    f"p conjunto RNDC = "
                    f"{bloques.iloc[0]['p_conjunto_HAC']:.5f}."
                ),
                (
                    f"p conjunto SIPSA = "
                    f"{bloques.iloc[1]['p_conjunto_HAC']:.5f}."
                ),
            ]

        except Exception as exc:
            contenido += [
                "",
                f"Modelo no estimable: {exc}",
            ]

    else:
        contenido += [
            "",
            (
                "Modelo pendiente por ausencia de "
                "series auténticas suficientes."
            ),
        ]

    contenido += [
        "",
        "## Alcance",
        (
            "No se identifican efectos causales "
            "ni se predicen cierres viales."
        ),
        "",
        "## Fuentes",
        f"https://github.com/{REPO_RNDC}",
        f"https://github.com/{REPO_SIPSA}",
    ]

    informe = "\n\n".join(contenido)

    st.markdown(informe)

    st.download_button(
        "Descargar informe Markdown",
        informe.encode("utf-8-sig"),
        file_name="informe_observatorio.md",
        mime="text/markdown",
    )

    exportar(
        DATA,
        "series_observatorio.csv",
    )


elif pagina == "💬 Chatbot":
    st.header(
        "Asistente numérico del observatorio"
    )

    st.caption(
        "No es RAG. La integración documental "
        "se desarrollará posteriormente."
    )

    if "dialogo" not in st.session_state:
        st.session_state.dialogo = []

    for item in st.session_state.dialogo:
        with st.chat_message(item["rol"]):
            st.markdown(item["texto"])

    pregunta = st.chat_input(
        "Pregunta por una variable, un mes AAAA-MM "
        "o su promedio"
    )

    if pregunta:
        q = normalizar(pregunta)

        if any(
            termino in q
            for termino in (
                "sipsa", "potrerillo", "mayorista"
            )
        ):
            variable = P

        elif any(
            termino in q
            for termino in ("rndc", "flete")
        ):
            variable = T

        elif "ictc" in q:
            variable = I

        else:
            variable = Y

        fecha = re.search(
            r"(20(?:18|19|2[0-4]))[- /](0?[1-9]|1[0-2])",
            q,
        )

        if any(
            termino in q
            for termino in (
                "causa", "causal", "bloqueo"
            )
        ):
            respuesta = (
                "Los modelos documentan asociaciones "
                "y no detectan bloqueos viales."
            )

        elif variable not in VARS:
            respuesta = (
                f"{NOMBRES[variable]} no está disponible "
                "en las series recuperadas."
            )

        elif fecha:
            periodo = pd.Period(
                f"{fecha.group(1)}-"
                f"{int(fecha.group(2)):02d}"
            )

            fila = DATA.loc[
                DATA["mes"] == periodo,
                variable,
            ]

            if (
                len(fila)
                and pd.notna(fila.iloc[0])
            ):
                respuesta = (
                    f"{NOMBRES[variable]} en {periodo}: "
                    f"{fila.iloc[0]:.5f}."
                )

            else:
                respuesta = (
                    "No existe una observación auténtica "
                    "para ese mes."
                )

        elif any(
            termino in q
            for termino in ("promedio", "media")
        ):
            respuesta = (
                f"Promedio observado de {NOMBRES[variable]}: "
                f"{DATA[variable].mean():.5f} "
                f"({DATA[variable].count()} meses)."
            )

        else:
            respuesta = (
                "Consulta un valor mensual (AAAA-MM), "
                "un promedio o la cobertura de una variable."
            )

        st.session_state.dialogo += [
            {
                "rol": "user",
                "texto": pregunta,
            },
            {
                "rol": "assistant",
                "texto": respuesta,
            },
        ]

        st.rerun()


else:
    st.header(
        "Derechos, fuentes y créditos"
    )

    st.markdown(
        "**Investigación:** Costos del transporte de carga, "
        "precios mayoristas de alimentos y brecha inflacionaria "
        "Pasto–Colombia, 2018–2024."
    )

    st.markdown(
        "**Autores del artículo:** Juan David Granja Miramag, "
        "Helmer Andrés Rivera Vallejos y "
        "Yeison Alexander Ortega Rosero. "
        "Universidad de Nariño."
    )

    st.markdown(
        "**Fuentes:** DANE (IPC, SIPSA e ICTC) y "
        "Ministerio de Transporte (RNDC). "
        "El índice regional de fletes es una "
        "construcción de los autores; GitHub "
        "funciona como respaldo y repositorio "
        "de productos procesados."
    )

    st.markdown(
        f"**RNDC:** https://github.com/{REPO_RNDC}"
    )

    st.markdown(
        f"**SIPSA:** https://github.com/{REPO_SIPSA}"
    )

    st.info(
        "No se realizan imputaciones. No se atribuye "
        "causalidad, no se generan alertas operativas "
        "y todavía no se implementa RAG."
    )


st.divider()

st.caption(
    "Observatorio Pasto–Nariño · "
    "Fuentes oficiales y derivados auditables · "
    "Evidencia mensual 2018–2024"
)
