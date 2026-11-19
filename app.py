
# app.py
# Instalar: pip install streamlit pandas numpy scipy requests plotly statsmodels scikit-learn openpyxl pyarrow ipython beautifulsoup4
# Ejecutar: streamlit run app.py

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey, acorr_ljungbox, breaks_cusumolsresid,
    het_breuschpagan, linear_reset,
)
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

st.set_page_config(
    page_title="Observatorio Pasto · Nariño",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg,#0b1220,#111a2e);
    color: #e2e8f0;
}
section[data-testid="stSidebar"] {
    background: #101b30;
    border-right: 1px solid #253850;
}
h1,h2,h3 {
    color: #9ce8f1 !important;
    letter-spacing: -.02em;
}
[data-testid="stMetric"] {
    background: #1b2a40;
    border: 1px solid #314760;
    border-radius: 13px;
    padding: 14px;
}
[data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 1px solid #314760;
}
.stButton button[kind="primary"] {
    background: linear-gradient(95deg,#087e9b,#6953bd);
    border: 0;
}
.hero {
    padding: 24px;
    border-radius: 18px;
    background: linear-gradient(110deg,#143047,#282045);
    border: 1px solid #334c66;
    margin-bottom: 20px;
}
.hero h2 {
    margin-top: 0;
    color: #a8f0f3 !important;
}
.hero p {
    color: #dce9f4;
    font-size: 1.06rem;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

Y = "BRECHA_INFLACION_PASTO_COLOMBIA_PP"
T = "VAR_FLETE_TORNQVIST_PCT"
P = "PM_ponderado_t"
I = "ICTC_MENSUAL_PCT"

M84 = pd.period_range("2018-01", "2024-12", freq="M")
M59 = pd.period_range("2020-02", "2024-12", freq="M")
M56 = pd.period_range("2020-05", "2024-12", freq="M")

MAIN = "JuanDavidgm2026/fletes-precios-inflacion-pasto"
SIPSA = "JuanDavidgm2026/SIPSA-PASTO"
TAG = "sipsa-dane-originales-v1"

NB_ID = "1Tu3ZzP1L8XlUejqbS_0eXgZ8QszqiMr7"
NB_SHA = "be21823d6980f6b0f092fd58ff808fa6056c0a550f01606ec9fb1d85276da98f"

CACHE_DIR = Path.cwd() / ".observatorio_cache"

NOM = {
    Y: "Brecha Pasto–Colombia (p. p.)",
    T: "RNDC regional (%)",
    P: "SIPSA ponderado (%)",
    I: "ICTC nacional (%)",
    "IPC_MENSUAL_PASTO_PCT": "Inflación mensual Pasto (%)",
    "IPC_MENSUAL_NACIONAL_PCT": "Inflación mensual Colombia (%)",
    "PM_no_ponderado_t": "SIPSA no ponderado (%)",
    "PM_ponderado_lagQ_t": "SIPSA con pesos rezagados (%)",
    "PM_100": "SIPSA canasta PM_100 (%)",
    "PM_95": "SIPSA canasta PM_95 (%)",
    "VAR_FLETE_FISHER_PCT": "RNDC Fisher (%)",
    "VAR_FLETE_CRUDO_PCT": "RNDC sin filtro (%)",
    "VAR_FLETE_SIN_2000_PCT": "RNDC sin valores de 2.000 (%)",
    "cobertura_indice_pct": "Cobertura SIPSA (%)",
    "n_productos_indice": "Productos SIPSA",
    "COB_VALOR_T_PCT": "Cobertura RNDC por valor (%)",
    "COB_VIAJES_T_PCT": "Cobertura RNDC por viajes (%)",
    "CELDAS_COMPARABLES": "Celdas RNDC comparables",
}

URLS = {
    "RNDC": [
        f"https://raw.githubusercontent.com/{MAIN}/main/data/"
        "variable2_flete_regional_modelo_2020_2024.csv"
    ],
    "IPC": [
        "https://www.dane.gov.co/files/operaciones/IPC/dic2024/"
        "anex-IPC-CiudadesMensuales-dic2024.xlsx",
        f"https://github.com/{SIPSA}/releases/download/{TAG}/"
        "anex-IPC-CiudadesMensuales-dic2024.xlsx",
    ],
    "ICTC": [
        "https://www.dane.gov.co/files/operaciones/ICTC/"
        "anex-ICTC-HistVariaContrib-dic2024.xlsx",
        f"https://github.com/{SIPSA}/releases/download/{TAG}/"
        "anex-ICTC-HistVariaContrib-dic2024.xlsx",
    ],
}

HASH = {
    "RNDC": "aaec180a88d32bbccbfb13b94be4784c9e9b4c6f23d9a93b7475b1acd1bd3891",
    "IPC": "3e584414f3f89c78f99ce92576a0715d3d51fac5d6dcd75addf84065e4cd4076",
    "ICTC": "6fe232548e47dbf9801b75c620209f579ebd0faa2a46fc30a9a305453878205a",
}

ART = {
    "N": 56,
    "RNDC0": 0.00913,
    "p_RNDC0": 0.03750,
    "p_RNDC_bloque": 0.14273,
    "p_SIPSA_bloque": 0.23749,
    "R2aj": 0.1033,
}


def etiqueta(x):
    return NOM.get(x, x)


def huella(raw):
    return hashlib.sha256(raw).hexdigest()


def normal(v):
    if pd.isna(v):
        return ""

    s = unicodedata.normalize("NFKD", str(v).strip().lower())

    return " ".join(
        "".join(
            c for c in s
            if not unicodedata.combining(c)
        ).split()
    )


def num(v):
    if pd.isna(v):
        return np.nan

    if isinstance(v, (int, float, np.number)):
        return float(v)

    return float(
        pd.to_numeric(
            str(v).strip().replace("%", "").replace(",", "."),
            errors="coerce",
        )
    )


def get(url, timeout=75):
    r = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "ObservatorioPasto/3.0"},
    )
    r.raise_for_status()
    return r.content


def fecha_mensual(s):
    if isinstance(s.dtype, pd.PeriodDtype):
        return s.dt.asfreq("M")

    v = s.astype("string").str.strip()

    if v.dropna().str.fullmatch(r"\d{6}").all():
        return pd.to_datetime(
            v, format="%Y%m", errors="coerce"
        ).dt.to_period("M")

    if v.dropna().str.fullmatch(r"\d{4}-\d{2}").all():
        return pd.to_datetime(
            v + "-01", errors="coerce"
        ).dt.to_period("M")

    return pd.to_datetime(
        s, errors="coerce"
    ).dt.to_period("M")


def exigir_meses(d, periodos, variables):
    if d["mes"].tolist() != list(periodos):
        raise ValueError(
            f"Cobertura distinta de {periodos[0]}–{periodos[-1]}"
        )

    if not np.isfinite(
        d[variables].to_numpy(dtype=float)
    ).all():
        raise ValueError("Serie incompleta o no finita")


def estandarizar(d, nombre):
    d = d.copy()

    d.columns = [
        str(c).lstrip("\ufeff").strip()
        for c in d.columns
    ]

    fecha = next(
        (
            c for c in [
                "mes", "FECHA", "fecha",
                "PERIODO", "periodo", "MES"
            ]
            if c in d
        ),
        None,
    )

    if fecha is None:
        raise ValueError(
            f"{nombre}: sin fecha reconocible"
        )

    d["mes"] = fecha_mensual(d[fecha])

    d = d[
        d["mes"].isin(M84)
    ].sort_values("mes")

    if d["mes"].duplicated().any():
        raise ValueError(
            f"{nombre}: meses duplicados"
        )

    cols = [
        c for c in NOM
        if c in d
    ]

    if not cols:
        raise ValueError(
            f"{nombre}: sin variables del estudio"
        )

    for c in cols:
        d[c] = pd.to_numeric(
            d[c], errors="coerce"
        )

    return d[
        ["mes"] + cols
    ].reset_index(drop=True)


def leer_tabla(raw, nombre):
    if nombre.lower().endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(raw))

    for cod in ["utf-8-sig", "latin-1"]:
        for sep in [",", ";"]:
            try:
                d = pd.read_csv(
                    io.BytesIO(raw),
                    encoding=cod,
                    sep=sep,
                )

                if d.shape[1] > 1:
                    return d

            except (UnicodeError, pd.errors.ParserError):
                pass

    raise ValueError("CSV/Parquet ilegible")


def ipc_dane(raw):
    a = pd.read_excel(
        io.BytesIO(raw),
        sheet_name="VarCiudades",
        header=None,
    )

    bloques = []

    for i, x in enumerate(a.iloc[:, 0]):
        match = re.search(
            r"variaciones mensuales\s+(20\d{2})",
            normal(x),
        )

        if match:
            bloques.append(
                (int(match.group(1)), i)
            )

    if [x[0] for x in bloques] != list(range(2017, 2025)):
        raise ValueError(
            "Hoja IPC distinta del anexo auditado"
        )

    meses = (
        "enero febrero marzo abril mayo junio julio agosto "
        "septiembre octubre noviembre diciembre"
    ).split()

    filas = []

    for k, (anio, ini) in enumerate(bloques):
        encabezados = [
            normal(a.iat[ini + 1, c])
            for c in range(1, 13)
        ]

        if encabezados != meses:
            raise ValueError(
                f"Encabezados IPC no reconocidos: {anio}"
            )

        fin = (
            bloques[k + 1][1]
            if k + 1 < len(bloques)
            else len(a)
        )

        pasto = [
            i for i in range(ini + 2, fin)
            if normal(a.iat[i, 0]) == "pasto"
        ]

        nacional = [
            i for i in range(ini + 2, fin)
            if normal(a.iat[i, 0]) == "total ipc"
        ]

        if len(pasto) != 1 or len(nacional) != 1:
            raise ValueError(
                f"Pasto/Total IPC ambiguos: {anio}"
            )

        for mes in range(1, 13):
            if anio < 2018:
                continue

            p = num(a.iat[pasto[0], mes])
            n = num(a.iat[nacional[0], mes])

            filas.append({
                "mes": pd.Period(f"{anio}-{mes:02d}"),
                "IPC_MENSUAL_PASTO_PCT": p,
                "IPC_MENSUAL_NACIONAL_PCT": n,
                Y: p - n,
            })

    d = pd.DataFrame(filas)

    exigir_meses(
        d,
        M84,
        [
            Y,
            "IPC_MENSUAL_PASTO_PCT",
            "IPC_MENSUAL_NACIONAL_PCT",
        ],
    )

    if not (
        np.isclose(
            d.iloc[-1]["IPC_MENSUAL_PASTO_PCT"], 0.47
        )
        and np.isclose(
            d.iloc[-1]["IPC_MENSUAL_NACIONAL_PCT"], 0.46
        )
    ):
        raise ValueError(
            "IPC: diciembre 2024 no coincide con el control del Colab"
        )

    return d


def ictc_dane(raw):
    a = pd.read_excel(
        io.BytesIO(raw),
        sheet_name="Anexo1",
        header=None,
    ).iloc[:, :3].copy()

    a.columns = [
        "anio_bruto",
        "mes_bruto",
        "valor_bruto",
    ]

    def anio(x):
        try:
            v = int(float(x))
            return float(v) if 2000 <= v <= 2100 else np.nan
        except (ValueError, TypeError):
            return np.nan

    a["anio"] = a["anio_bruto"].map(anio).ffill()

    mapa = {
        m: i + 1
        for i, m in enumerate(
            "enero febrero marzo abril mayo junio julio agosto "
            "septiembre octubre noviembre diciembre".split()
        )
    }

    a["nmes"] = a["mes_bruto"].map(normal).map(mapa)
    a[I] = a["valor_bruto"].map(num)

    a = a[
        a["anio"].between(2018, 2024)
        & a["nmes"].notna()
        & a[I].notna()
    ].copy()

    a["mes"] = pd.PeriodIndex(
        year=a["anio"].astype(int),
        month=a["nmes"].astype(int),
        freq="M",
    )

    d = a[
        ["mes", I]
    ].sort_values("mes").reset_index(drop=True)

    exigir_meses(d, M84, [I])

    if not (
        np.isclose(d.iloc[0][I], 0.70)
        and np.isclose(d.iloc[1][I], 0.83)
    ):
        raise ValueError(
            "ICTC: controles de enero/febrero de 2018 no coinciden"
        )

    return d


def fuente_oficial(nombre):
    errores = []

    for url in URLS[nombre]:
        try:
            raw = get(url)

            if huella(raw) != HASH[nombre]:
                raise ValueError("SHA-256 diferente")

            parsers = {
                "IPC": ipc_dane,
                "ICTC": ictc_dane,
                "RNDC": lambda x: estandarizar(
                    leer_tabla(x, "rndc.csv"), "RNDC"
                ),
            }

            d = parsers[nombre](raw)

            if nombre == "RNDC":
                exigir_meses(d, M59, [T])

            registro = {
                "archivo": nombre,
                "fuente": url,
                "sha256": huella(raw),
                "control": "SHA verificado",
            }

            return d, registro

        except Exception as e:
            errores.append(f"{url}: {e}")

    raise RuntimeError(" | ".join(errores))


def descubrir_procesados():
    claves = (
        "sipsa_indicadores",
        "indicadores_mensuales",
        "base_integrada",
        "base_econometrica",
        "base_estimacion",
        "pm_ponderado",
        "cobertura_alternativas",
    )

    salidas = []
    errores = []
    vistos = set()

    for repo in [SIPSA, MAIN]:
        try:
            r = requests.get(
                f"https://api.github.com/repos/{repo}",
                timeout=20,
            )
            r.raise_for_status()

            rama = r.json().get(
                "default_branch", "main"
            )

            r = requests.get(
                f"https://api.github.com/repos/{repo}/"
                f"git/trees/{rama}?recursive=1",
                timeout=30,
            )
            r.raise_for_status()

            for obj in r.json().get("tree", []):
                ruta = obj.get("path", "")
                archivo = Path(ruta).name.lower()

                if obj.get("type") != "blob":
                    continue

                if not archivo.endswith((".csv", ".parquet")):
                    continue

                if not any(k in archivo for k in claves):
                    continue

                url = (
                    f"https://raw.githubusercontent.com/"
                    f"{repo}/{rama}/{ruta}"
                )

                if url not in vistos:
                    salidas.append((ruta, url))
                    vistos.add(url)

        except Exception as e:
            errores.append(
                f"GitHub {repo}: {e}"
            )

        try:
            r = requests.get(
                f"https://api.github.com/repos/{repo}/releases",
                timeout=25,
            )
            r.raise_for_status()

            for rel in r.json()[:10]:
                for asset in rel.get("assets", []):
                    archivo = asset["name"].lower()

                    if (
                        archivo.endswith((".csv", ".parquet"))
                        and any(k in archivo for k in claves)
                    ):
                        url = asset["browser_download_url"]

                        if url not in vistos:
                            salidas.append(
                                (asset["name"], url)
                            )
                            vistos.add(url)

        except Exception as e:
            errores.append(
                f"Releases {repo}: {e}"
            )

    return salidas, errores


@st.cache_data(
    ttl=3600,
    show_spinner="Conectando con DANE y respaldos verificados de GitHub…",
)
def cargar_base():
    base = pd.DataFrame({
        "mes": M84
    })

    bitacora = []
    incidencias = []
    series = []

    for nombre in ["RNDC", "IPC", "ICTC"]:
        try:
            d, registro = fuente_oficial(nombre)

            series.append((nombre, d))
            bitacora.append(registro)

        except Exception as e:
            incidencias.append(
                f"{nombre}: {str(e)[:380]}"
            )

    candidatos, problemas = descubrir_procesados()
    incidencias.extend(problemas)

    for nombre, url in candidatos:
        try:
            raw = get(url)

            d = estandarizar(
                leer_tabla(raw, nombre),
                nombre,
            )

            series.append((nombre, d))

            bitacora.append({
                "archivo": nombre,
                "fuente": url,
                "sha256": huella(raw),
                "control": "SHA calculado; sin hash de referencia",
            })

        except Exception as e:
            incidencias.append(
                f"{nombre}: {str(e)[:220]}"
            )

    local = (
        CACHE_DIR
        / "SIPSA_AUDITORIA_RESULTADOS"
        / "SIPSA_INDICADORES_MENSUALES_2018_2024_CORREGIDO.csv"
    )

    if local.exists():
        try:
            raw = local.read_bytes()

            d = estandarizar(
                leer_tabla(raw, local.name),
                local.name,
            )

            exigir_meses(d, M84, [P])

            series.append(
                ("SIPSA reconstruido", d)
            )

            bitacora.append({
                "archivo": local.name,
                "fuente": "Fase SIPSA verificada del Colab",
                "sha256": huella(raw),
                "control": "Salida local verificada por calendario",
            })

        except Exception as e:
            incidencias.append(
                f"SIPSA reconstruido: {e}"
            )

    bloqueadas = set()
    procedencia = {}

    for nombre, d in series:
        d = d.set_index("mes").reindex(M84)

        for c in d.columns:
            if c in bloqueadas:
                continue

            nuevo = pd.to_numeric(
                d[c],
                errors="coerce",
            ).reset_index(drop=True)

            if c not in base:
                base[c] = nuevo.to_numpy()
                procedencia[c] = nombre
                continue

            anterior = pd.to_numeric(
                base[c], errors="coerce"
            )

            comunes = anterior.notna() & nuevo.notna()

            if comunes.any() and not np.allclose(
                anterior[comunes],
                nuevo[comunes],
                rtol=1e-7,
                atol=1e-8,
            ):
                incidencias.append(
                    f"Incompatibilidad en {c}: "
                    f"{procedencia[c]} y {nombre}"
                )

                base[c] = np.nan
                bloqueadas.add(c)

            else:
                base[c] = anterior.combine_first(nuevo)

    return base, pd.DataFrame(bitacora), incidencias


def reconstruir_sipsa():
    # Ejecuta únicamente la fase cuyo SHA coincide con el Colab auditado.
    endpoints = [
        f"https://drive.google.com/uc?export=download&id={NB_ID}",
        (
            "https://drive.usercontent.google.com/download?"
            f"id={NB_ID}&export=download&confirm=t"
        ),
    ]

    errores = []

    for url in endpoints:
        try:
            nb = json.loads(
                get(url, timeout=90).decode("utf-8-sig")
            )

            if (
                not isinstance(nb.get("cells"), list)
                or len(nb["cells"]) < 3
            ):
                raise ValueError(
                    "Notebook público no disponible"
                )

            script = "".join(
                nb["cells"][2]["source"]
            )

            if huella(script.encode()) != NB_SHA:
                raise ValueError(
                    "La versión del notebook cambió; ejecución bloqueada"
                )

            CACHE_DIR.mkdir(exist_ok=True)

            rutas = {
                "/content/SIPSA_DANE": str(
                    CACHE_DIR / "SIPSA_DANE"
                ),
                "/content/SIPSA_GITHUB_CACHE": str(
                    CACHE_DIR / "SIPSA_GITHUB_CACHE"
                ),
                "/content/SIPSA_AUDITORIA_RESULTADOS": str(
                    CACHE_DIR / "SIPSA_AUDITORIA_RESULTADOS"
                ),
            }

            for original, nuevo in rutas.items():
                script = script.replace(
                    original, nuevo
                )

            ruta = CACHE_DIR / "fase_sipsa_auditada.py"

            ruta.write_text(
                script,
                encoding="utf-8",
            )

            resultado = subprocess.run(
                [sys.executable, str(ruta)],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=3600,
            )

            if resultado.returncode:
                raise RuntimeError(
                    (resultado.stderr or resultado.stdout)[-1500:]
                )

            final = (
                CACHE_DIR
                / "SIPSA_AUDITORIA_RESULTADOS"
                / "SIPSA_INDICADORES_MENSUALES_2018_2024_CORREGIDO.csv"
            )

            if not final.exists():
                raise ValueError(
                    "La fase no generó el indicador mensual"
                )

            exigir_meses(
                estandarizar(
                    pd.read_csv(final),
                    final.name,
                ),
                M84,
                [P],
            )

            return final

        except Exception as e:
            errores.append(str(e)[:300])

    raise RuntimeError(" | ".join(errores))


def muestra(d, variables, periodos=M59):
    faltantes = [
        c for c in variables
        if c not in d
    ]

    if faltantes:
        raise ValueError(
            "Variables no descargadas: " + ", ".join(faltantes)
        )

    z = d[
        d["mes"].isin(periodos)
    ][["mes"] + variables].copy().reset_index(drop=True)

    exigir_meses(
        z, periodos, variables
    )

    return z


def matriz_ardl(
    d,
    L=2,
    ar=True,
    transporte=T,
    precio=P,
):
    z = muestra(
        d,
        [Y, transporte, precio],
    )

    x = z[["mes", Y]].copy()
    columnas = []

    if ar:
        x["B_L1"] = z[Y].shift(1)
        columnas.append("B_L1")

    for columna, prefijo in [
        (transporte, "T"),
        (precio, "P"),
    ]:
        for lag in range(L + 1):
            nombre = f"{prefijo}_L{lag}"

            x[nombre] = z[columna].shift(lag)
            columnas.append(nombre)

    x = x[
        x["mes"].isin(M56)
    ].dropna().reset_index(drop=True)

    exigir_meses(
        x,
        M56,
        [Y] + columnas,
    )

    return x, columnas


def ardl(
    d,
    L=2,
    ar=True,
    hac=3,
    transporte=T,
    precio=P,
    excluir=None,
    trimestres=False,
):
    z, columnas = matriz_ardl(
        d, L, ar, transporte, precio
    )

    if excluir:
        z = z[
            z["mes"] != pd.Period(excluir)
        ].copy()

    if trimestres:
        q = (
            z["mes"].dt.month.sub(1) // 3
        ).add(1)

        for j in (2, 3, 4):
            name = f"Q{j}"

            z[name] = (q == j).astype(int)
            columnas.append(name)

    X = sm.add_constant(
        z[columnas].astype(float),
        has_constant="add",
    )

    if len(z) < X.shape[1] + 10:
        raise ValueError(
            "Grados de libertad insuficientes"
        )

    mco = sm.OLS(
        z[Y], X
    ).fit()

    hacfit = sm.OLS(
        z[Y], X
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac},
        use_t=True,
    )

    return hacfit, mco, z, X


def tabla_coef(m):
    lim = m.conf_int()

    return pd.DataFrame({
        "Regresor": m.params.index,
        "Coeficiente": m.params.values,
        "Error HAC": m.bse.values,
        "p HAC": m.pvalues.values,
        "IC 95 % inf": lim.iloc[:, 0].values,
        "IC 95 % sup": lim.iloc[:, 1].values,
    })


def bloques(m, L=2):
    filas = []

    for etiqueta_bloque, pref in [
        ("RNDC / transporte", "T"),
        ("SIPSA / precios", "P"),
    ]:
        cols = [
            f"{pref}_L{i}"
            for i in range(L + 1)
        ]

        R = np.zeros(
            (len(cols), len(m.params))
        )

        for i, c in enumerate(cols):
            R[i, m.params.index.get_loc(c)] = 1

        joint = m.wald_test(
            R,
            use_f=True,
            scalar=True,
        )

        summed = m.t_test(
            R.sum(axis=0)
        )

        ci = summed.conf_int()[0]

        filas.append({
            "Bloque": etiqueta_bloque,
            "p conjunta HAC": float(joint.pvalue),
            "Suma": float(m.params[cols].sum()),
            "IC suma inf": float(ci[0]),
            "IC suma sup": float(ci[1]),
        })

    return pd.DataFrame(filas)


def diagnosticos(mco, z, X):
    e = mco.resid

    valores = {
        "Jarque–Bera p": float(jarque_bera(e)[1]),
        "Ljung–Box 6 p": float(
            acorr_ljungbox(
                e,
                lags=[6],
                return_df=True,
            )["lb_pvalue"].iloc[0]
        ),
        "Ljung–Box 12 p": float(
            acorr_ljungbox(
                e,
                lags=[12],
                return_df=True,
            )["lb_pvalue"].iloc[0]
        ),
        "Breusch–Godfrey p": float(
            acorr_breusch_godfrey(
                mco, nlags=3
            )[1]
        ),
        "Breusch–Pagan p": float(
            het_breuschpagan(e, X)[1]
        ),
        "CUSUM p": float(
            breaks_cusumolsresid(
                e,
                ddof=X.shape[1],
            )[1]
        ),
        "Condición X": float(
            np.linalg.cond(X)
        ),
    }

    try:
        valores["RESET p"] = float(
            linear_reset(
                mco,
                power=2,
                use_f=True,
            ).pvalue
        )
    except Exception:
        valores["RESET p"] = np.nan

    resid = z[["mes", Y]].copy()

    resid["Ajuste"] = mco.fittedvalues.to_numpy()
    resid["Residuo"] = e.to_numpy()
    resid["Cook"] = np.asarray(
        OLSInfluence(mco).cooks_distance[0]
    )

    vif = pd.DataFrame([
        {
            "Regresor": c,
            "VIF": variance_inflation_factor(
                X.to_numpy(float), i
            ),
        }
        for i, c in enumerate(X.columns)
        if c != "const"
    ])

    return (
        pd.DataFrame([valores]),
        resid,
        vif,
    )


def estacionariedad(d, cols):
    filas = []

    for c in cols:
        try:
            z = muestra(d, [c])

            a = adfuller(
                z[c],
                autolag="AIC",
            )

            k = kpss(
                z[c],
                regression="c",
                nlags="auto",
            )

            filas.append({
                "Serie": etiqueta(c),
                "N": len(z),
                "ADF": a[0],
                "p ADF": a[1],
                "KPSS": k[0],
                "p KPSS (tab.)": k[1],
                "Compatible I(0)": bool(
                    a[1] < 0.05 and k[1] >= 0.05
                ),
            })

        except Exception as e:
            filas.append({
                "Serie": etiqueta(c),
                "Estado": str(e)[:130],
            })

    return pd.DataFrame(filas)


def robustez(d):
    especificaciones = [
        (
            f"{tipo} L{l}",
            {
                "L": l,
                "ar": tipo == "ARDL",
            },
        )
        for tipo in ("DL", "ARDL")
        for l in range(4)
    ]

    especificaciones += [
        (f"HAC {h}", {"hac": h})
        for h in (1, 6)
    ]

    especificaciones += [
        (
            "Sin febrero 2023",
            {"excluir": "2023-02"},
        ),
        (
            "Estacionalidad trimestral",
            {"trimestres": True},
        ),
    ]

    for c in [
        "PM_no_ponderado_t",
        "PM_ponderado_lagQ_t",
        "PM_100",
        "PM_95",
    ]:
        if c in d and d[c].notna().sum() >= 59:
            especificaciones.append(
                (f"SIPSA {c}", {"precio": c})
            )

    if I in d and d[I].notna().sum() == 84:
        especificaciones.append(
            (
                "ICTC reemplaza RNDC",
                {"transporte": I},
            )
        )

    filas = []

    for name, params in especificaciones:
        try:
            m, o, z, _ = ardl(
                d, **params
            )

            b = bloques(
                m, params.get("L", 2)
            )

            filas.append({
                "Especificación": name,
                "N": len(z),
                "R² aj.": o.rsquared_adj,
                "AIC": o.aic,
                "BIC": o.bic,
                "p RNDC/ICTC": b.iloc[0]["p conjunta HAC"],
                "p SIPSA": b.iloc[1]["p conjunta HAC"],
                "Suma transporte": b.iloc[0]["Suma"],
                "Estado": "OK",
            })

        except Exception as e:
            filas.append({
                "Especificación": name,
                "Estado": str(e)[:140],
            })

    return pd.DataFrame(filas)


def ventanas(d, ancho=36):
    z, cols = matriz_ardl(d)
    filas = []

    for i in range(len(z) - ancho + 1):
        v = z.iloc[i:i + ancho]

        X = sm.add_constant(
            v[cols],
            has_constant="add",
        )

        m = sm.OLS(
            v[Y], X
        ).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 3},
            use_t=True,
        )

        b = bloques(m)

        filas.append({
            "Inicio": str(v["mes"].iloc[0]),
            "Fin": str(v["mes"].iloc[-1]),
            "Suma RNDC": b.iloc[0]["Suma"],
            "p conjunta RNDC": b.iloc[0]["p conjunta HAC"],
            "Suma SIPSA": b.iloc[1]["Suma"],
        })

    return pd.DataFrame(filas)


def huber(d):
    z, cols = matriz_ardl(d)

    X = sm.add_constant(
        z[cols],
        has_constant="add",
    )

    a = sm.OLS(
        z[Y], X
    ).fit()

    b = sm.RLM(
        z[Y],
        X,
        M=sm.robust.norms.HuberT(),
    ).fit()

    return pd.DataFrame([
        {
            "Bloque": label,
            "Suma MCO": a.params[
                [f"{p}_L{i}" for i in range(3)]
            ].sum(),
            "Suma Huber": b.params[
                [f"{p}_L{i}" for i in range(3)]
            ].sum(),
        }
        for label, p in [
            ("RNDC", "T"),
            ("SIPSA", "P"),
        ]
    ])


def trayectorias(m, var, h=6, shock=1.0):
    cols = list(m.params.index)
    cov = m.cov_params().to_numpy(float)

    rho = float(
        m.params.get("B_L1", 0)
    )

    pos = (
        cols.index("B_L1")
        if "B_L1" in cols
        else None
    )

    g = np.zeros(len(cols))
    r = 0
    filas = []

    pref = "T" if var == T else "P"

    for paso in range(h + 1):
        prev = r
        old = g.copy()
        nombre = f"{pref}_L{paso}"

        r = (
            rho * prev
            + float(m.params.get(nombre, 0))
        )

        g = rho * old

        if pos is not None:
            g[pos] += prev

        if nombre in cols:
            g[cols.index(nombre)] += 1

        error = np.sqrt(
            max(float(g @ cov @ g), 0)
        )

        filas.append({
            "Mes": paso,
            "Respuesta": shock * r,
            "IC inf": shock * r - 1.96 * abs(shock) * error,
            "IC sup": shock * r + 1.96 * abs(shock) * error,
        })

    return pd.DataFrame(filas)


def pca_clusters(d, cols, k):
    z = muestra(d, cols)

    X = StandardScaler().fit_transform(
        z[cols]
    )

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)

    etiquetas = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    ).fit_predict(X)

    puntos = pd.DataFrame({
        "Mes": z["mes"].astype(str),
        "PC1": scores[:, 0],
        "PC2": scores[:, 1],
        "Grupo": etiquetas.astype(str),
    })

    cargas = pd.DataFrame(
        pca.components_.T,
        index=[etiqueta(c) for c in cols],
        columns=["PC1", "PC2"],
    )

    return (
        puntos,
        cargas,
        pca.explained_variance_ratio_,
    )


def var_exploratorio(d, cols, lags, h):
    z = muestra(d, cols)

    for c in cols:
        a = adfuller(
            z[c],
            autolag="AIC",
        )[1]

        k = kpss(
            z[c],
            regression="c",
            nlags="auto",
        )[1]

        if not (a < 0.05 and k >= 0.05):
            raise ValueError(
                "VAR en niveles exige evidencia I(0) "
                "para todas las series"
            )

    m = VAR(
        z[cols]
    ).fit(lags)

    if not m.is_stable():
        raise ValueError("VAR inestable")

    return m, m.irf(h).irfs


def cointegracion(d, cols):
    z = muestra(d, cols)

    test = pd.DataFrame([
        {
            "Serie": etiqueta(c),
            "ADF p nivel": adfuller(
                z[c],
                autolag="AIC",
            )[1],
            "ADF p diferencia": adfuller(
                z[c].diff().dropna(),
                autolag="AIC",
            )[1],
        }
        for c in cols
    ])

    condicion = (
        (test["ADF p nivel"] >= 0.05)
        & (test["ADF p diferencia"] < 0.05)
    )

    if not condicion.all():
        return test, None

    j = coint_johansen(
        z[cols],
        det_order=0,
        k_ar_diff=1,
    )

    return test, pd.DataFrame({
        "Traza": j.lr1,
        "Crítico 95 %": j.cvt[:, 1],
    })


def ml_cronologico(d, seleccion):
    z = muestra(
        d,
        [Y, T, P],
    )

    X = pd.DataFrame({
        "mes": z["mes"],
        "objetivo": z[Y],
    })

    for c in [Y, T, P]:
        for lag in [1, 2]:
            X[f"{c}_L{lag}"] = z[c].shift(lag)

    X = X.dropna().reset_index(drop=True)

    cols = [
        c for c in X
        if c not in ["mes", "objetivo"]
    ]

    inicio = len(X) - 26
    filas = []

    modelos = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(
            alpha=0.01,
            max_iter=10000,
        ),
        "SVR": SVR(),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(12,),
            max_iter=500,
            random_state=42,
        ),
    }

    for i in range(inicio, len(X)):
        ent = X.iloc[:i]
        pr = X.iloc[[i]]

        tr = ent[cols].to_numpy(float)
        te = pr[cols].to_numpy(float)

        y = ent["objetivo"].to_numpy(float)

        sc = StandardScaler().fit(tr)

        pred = {
            "Media histórica": float(y.mean())
        }

        ar1 = sm.OLS(
            y,
            sm.add_constant(
                ent[[f"{Y}_L1"]],
                has_constant="add",
            ),
        ).fit()

        pred["AR(1)"] = float(
            ar1.predict(
                sm.add_constant(
                    pr[[f"{Y}_L1"]],
                    has_constant="add",
                )
            ).iloc[0]
        )

        arx = sm.OLS(
            y,
            sm.add_constant(
                tr,
                has_constant="add",
            ),
        ).fit()

        pred["ARX"] = float(
            arx.predict(
                sm.add_constant(
                    te,
                    has_constant="add",
                )
            )[0]
        )

        for nombre, m in modelos.items():
            if nombre in seleccion:
                m.fit(
                    sc.transform(tr),
                    y,
                )

                pred[nombre] = float(
                    m.predict(
                        sc.transform(te)
                    )[0]
                )

        for nombre, valor in pred.items():
            if nombre in seleccion:
                filas.append({
                    "Mes": str(pr["mes"].iloc[0]),
                    "Modelo": nombre,
                    "Real": float(
                        pr["objetivo"].iloc[0]
                    ),
                    "Predicho": valor,
                })

    resultados = pd.DataFrame(filas)

    resumen = pd.DataFrame([
        {
            "Modelo": nombre,
            "N": len(g),
            "RMSE": float(
                np.sqrt(
                    mean_squared_error(
                        g.Real, g.Predicho
                    )
                )
            ),
            "MAE": float(
                mean_absolute_error(
                    g.Real, g.Predicho
                )
            ),
        }
        for nombre, g in resultados.groupby("Modelo")
    ]).sort_values("RMSE")

    return resumen, resultados, X, cols, inicio


def xai_permutacion(tabla, cols, inicio, nombre):
    X = tabla[cols].to_numpy(float)
    y = tabla["objetivo"].to_numpy(float)

    sc = StandardScaler().fit(
        X[:inicio]
    )

    modelos = {
        "Ridge": Ridge(),
        "Lasso": Lasso(
            alpha=0.01,
            max_iter=10000,
        ),
        "SVR": SVR(),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(12,),
            max_iter=500,
            random_state=42,
        ),
    }

    m = modelos[nombre].fit(
        sc.transform(X[:inicio]),
        y[:inicio],
    )

    res = permutation_importance(
        m,
        sc.transform(X[inicio:]),
        y[inicio:],
        n_repeats=12,
        random_state=42,
        scoring="neg_mean_squared_error",
    )

    return pd.DataFrame({
        "Regresor": cols,
        "Importancia": res.importances_mean,
        "Desviación": res.importances_std,
    }).sort_values("Importancia")


def semaforo(d):
    z = muestra(d, [T, P])
    filas = []

    for i in range(12, len(z)):
        anterior = z.iloc[:i]
        actual = z.iloc[i]

        uT = anterior[T].quantile(0.90)
        uP = anterior[P].quantile(0.90)

        n = (
            int(actual[T] > uT)
            + int(actual[P] > uP)
        )

        filas.append({
            "Mes": str(actual.mes),
            "RNDC": actual[T],
            "SIPSA": actual[P],
            "Umbral RNDC": uT,
            "Umbral SIPSA": uP,
            "Semáforo": (
                "Verde",
                "Amarillo",
                "Rojo",
            )[n],
        })

    return pd.DataFrame(filas)


def grafico(fig, h=430):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1729",
        plot_bgcolor="#101b2e",
        font=dict(color="#dbeafe"),
        height=h,
        margin=dict(
            t=58, l=24, r=24, b=30
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    return fig


def serie_grafico(d, cols, titulo):
    tmp = d[
        ["mes"] + cols
    ].copy()

    tmp["mes"] = tmp["mes"].astype(str)

    if len(cols) == 1:
        return grafico(
            px.line(
                tmp,
                x="mes",
                y=cols[0],
                title=titulo,
                labels={cols[0]: etiqueta(cols[0])},
            )
        )

    fig = go.Figure()

    for c in cols:
        s = tmp[
            ["mes", c]
        ].dropna()

        fig.add_trace(
            go.Scatter(
                x=s.mes,
                y=s[c],
                mode="lines",
                name=etiqueta(c),
            )
        )

    fig.update_layout(
        title=titulo,
        yaxis_title="Unidades originales; comparar con precaución",
    )

    return grafico(fig)


def exportar(d, nombre):
    d = d.copy()

    if "mes" in d:
        d["mes"] = d["mes"].astype(str)

    st.download_button(
        "⬇️ Descargar tabla CSV",
        data=d.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name=nombre,
        mime="text/csv",
    )


def necesita(d, cols):
    faltan = [
        c for c in cols
        if c not in d
        or d[c].notna().sum() == 0
    ]

    if faltan:
        st.info(
            "No están disponibles estas series: "
            + ", ".join(
                etiqueta(c) for c in faltan
            )
        )

    return not faltan


with st.sidebar:
    st.markdown("### 📊 OBSERVATORIO")
    st.caption(
        "Pasto · El Potrerillo · Transporte regional"
    )

    pagina = st.radio(
        "Explorar",
        [
            "🏠 Inicio",
            "📚 Datos y muestras",
            "📊 Análisis económico",
            "📑 Generador de informes",
            "💬 Asistente estadístico",
            "© Fuentes y créditos",
        ],
    )

    st.divider()

    periodos = [
        str(v) for v in M84
    ]

    desde, hasta = st.select_slider(
        "Periodo de visualización",
        options=periodos,
        value=(periodos[0], periodos[-1]),
    )

    if st.button("🔄 Actualizar datos públicos"):
        cargar_base.clear()
        st.rerun()


DATA, AUDITORIA, INCIDENCIAS = cargar_base()

DISP = [
    c for c in NOM
    if c in DATA
    and DATA[c].notna().any()
]

VARS = [
    c for c in [Y, T, P, I]
    if c in DISP
]

VISTA = DATA[
    DATA["mes"].between(
        pd.Period(desde),
        pd.Period(hasta),
    )
].copy()


if pagina == "🏠 Inicio":
    st.title(
        "Observatorio Económico de Pasto y Nariño"
    )

    st.markdown("""
<div class='hero'>
<h2>¡Bienvenidos al observatorio! 👋</h2>
<p>
Explora con gráficos interactivos cómo evolucionaron
los fletes regionales, los precios mayoristas de
El Potrerillo y la brecha inflacionaria entre Pasto
y Colombia durante 2018–2024.
Usa el menú para comparar indicadores, analizar
relaciones, probar modelos y descargar resultados.
</p>
</div>
""", unsafe_allow_html=True)

    tarjetas = st.columns(4)

    for col, variable in zip(
        tarjetas, [Y, P, T, I]
    ):
        n = (
            int(DATA[variable].count())
            if variable in DATA
            else 0
        )

        col.metric(
            etiqueta(variable),
            f"{n} meses" if n else "Pendiente",
        )

    a, b = st.columns([1.55, 1])

    with a:
        st.subheader("Pulso mensual de la región")

        selec = (
            st.selectbox(
                "Indicador",
                VARS,
                format_func=etiqueta,
            )
            if VARS else None
        )

        if selec:
            serie_grafico(
                VISTA,
                [selec],
                etiqueta(selec),
            )

    with b:
        st.subheader("Panorama de la muestra")

        cobertura = pd.DataFrame([
            {
                "Serie": etiqueta(c),
                "Meses": int(DATA[c].count()),
                "Participación %": (
                    100 * DATA[c].count() / 84
                ),
            }
            for c in VARS
        ])

        if not cobertura.empty:
            grafico(
                px.bar(
                    cobertura,
                    x="Meses",
                    y="Serie",
                    orientation="h",
                    title="Cobertura disponible",
                ),
                420,
            )

    if len(VARS) > 1:
        st.subheader("Comparador de evolución")

        seleccion = st.multiselect(
            "Compara series",
            VARS,
            default=VARS[:min(3, len(VARS))],
            format_func=etiqueta,
        )

        if seleccion:
            normalizado = st.toggle(
                "Mostrar en desviaciones estándar (solo gráfica)",
                value=True,
            )

            comp = VISTA[
                ["mes"] + seleccion
            ].copy()

            if normalizado:
                for c in seleccion:
                    if comp[c].std() > 0:
                        comp[c] = (
                            comp[c] - comp[c].mean()
                        ) / comp[c].std()

            serie_grafico(
                comp,
                seleccion,
                "Evolución comparada",
            )

    st.subheader("Explora por herramienta")

    col1, col2, col3 = st.columns(3)

    col1.markdown(
        "**📊 Estadística y econometría**\n\n"
        "EDA, ADF/KPSS, ARDL, HAC, diagnósticos y robustez."
    )

    col2.markdown(
        "**🧩 Analítica avanzada**\n\n"
        "PCA, K-Means, VAR, Machine Learning y explicabilidad."
    )

    col3.markdown(
        "**🎯 Aplicaciones**\n\n"
        "Escenarios, episodio 2021, consultas e informes exportables."
    )


elif pagina == "📚 Datos y muestras":
    st.title("Datos y muestras")

    cob = pd.DataFrame([
        {
            "Indicador": etiqueta(c),
            "Observaciones": int(DATA[c].count()),
            "Desde": str(
                DATA.loc[
                    DATA[c].notna(), "mes"
                ].min()
            ),
            "Hasta": str(
                DATA.loc[
                    DATA[c].notna(), "mes"
                ].max()
            ),
        }
        for c in DISP
    ])

    st.dataframe(
        cob,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Explorador de observaciones")

    elegidos = st.multiselect(
        "Variables",
        DISP,
        default=VARS,
        format_func=etiqueta,
    )

    if elegidos:
        tabla = VISTA[
            ["mes"] + elegidos
        ].copy()

        tabla["mes"] = tabla["mes"].astype(str)

        st.dataframe(
            tabla,
            use_container_width=True,
            hide_index=True,
            height=370,
        )

        exportar(
            tabla,
            "observaciones_pasto.csv",
        )

    st.subheader("Procedencia y verificación")

    st.dataframe(
        AUDITORIA,
        use_container_width=True,
        hide_index=True,
    )

    if INCIDENCIAS:
        with st.expander(
            f"Detalles técnicos de carga ({len(INCIDENCIAS)})"
        ):
            for i in INCIDENCIAS:
                st.write(i)

    if P not in DISP:
        st.subheader(
            "Preparar el indicador SIPSA desde sus archivos originales"
        )

        st.write(
            "La app ya buscó productos mensuales publicados en GitHub. "
            "Si no existen, puede ejecutar la fase exacta del Colab auditado, "
            "con descarga de respaldos oficiales. Es una reconstrucción "
            "de mayor tamaño que se guarda en caché."
        )

        if st.button(
            "Reconstruir SIPSA automáticamente",
            type="primary",
        ):
            try:
                with st.spinner(
                    "Reconstruyendo SIPSA desde el procedimiento del proyecto…"
                ):
                    salida = reconstruir_sipsa()

                st.success(
                    f"Indicador generado: {salida.name}"
                )

                cargar_base.clear()
                st.rerun()

            except Exception as e:
                st.error(
                    f"Reconstrucción no finalizada: {e}"
                )


elif pagina == "📊 Análisis económico":
    st.title(
        "Laboratorio estadístico y econométrico"
    )

    tab = st.tabs([
        "📊 EDA",
        "📈 Estacionariedad",
        "🧮 ARDL + HAC",
        "🩺 Diagnósticos",
        "🛡️ Robustez",
        "🧩 PCA / K-Means",
        "🔁 VAR / VECM",
        "🤖 ML y XAI",
        "🎛️ Simulador",
        "📍 Focus Pasto",
    ])

    with tab[0]:
        elegir = st.multiselect(
            "Indicadores para explorar",
            DISP,
            default=VARS,
            format_func=etiqueta,
        )

        if elegir:
            st.dataframe(
                VISTA[elegir].describe().T,
                use_container_width=True,
            )

            serie_grafico(
                VISTA,
                [elegir[0]],
                "Dinámica del indicador",
            )

            if len(elegir) > 1:
                st.markdown(
                    "**Matriz de correlación**"
                )

                tipo = st.radio(
                    "Método",
                    ["pearson", "spearman"],
                    horizontal=True,
                )

                corr = VISTA[elegir].corr(
                    method=tipo,
                    min_periods=12,
                )

                grafico(
                    px.imshow(
                        corr,
                        text_auto=".2f",
                        zmin=-1,
                        zmax=1,
                        color_continuous_scale="RdBu_r",
                        title="Asociación bivariada",
                    )
                )

                a, b = st.columns(2)

                with a:
                    x = st.selectbox(
                        "Eje X",
                        elegir,
                        format_func=etiqueta,
                    )

                with b:
                    y = st.selectbox(
                        "Eje Y",
                        elegir,
                        index=1,
                        format_func=etiqueta,
                    )

                puntos = VISTA[
                    [x, y]
                ].dropna()

                grafico(
                    px.scatter(
                        puntos,
                        x=x,
                        y=y,
                        title="Dispersión de variables",
                    )
                )

            grafico(
                px.box(
                    VISTA[elegir].rename(
                        columns=NOM
                    ),
                    points="outliers",
                    title="Dispersión y valores extremos",
                )
            )

            exportar(
                VISTA[["mes"] + elegir],
                "eda_series.csv",
            )

    with tab[1]:
        if VARS:
            res = estacionariedad(
                DATA, VARS
            )

            st.dataframe(
                res,
                use_container_width=True,
                hide_index=True,
            )

            exportar(
                res,
                "adf_kpss.csv",
            )

            a = st.selectbox(
                "Correlograma de la variable",
                VARS,
                format_func=etiqueta,
            )

            try:
                z = muestra(DATA, [a])
                s = z[a].to_numpy()

                lagmax = min(
                    18,
                    len(s) // 3,
                )

                acf = sm.tsa.stattools.acf(
                    s,
                    nlags=lagmax,
                    fft=False,
                )

                grafico(
                    px.bar(
                        x=np.arange(len(acf)),
                        y=acf,
                        labels={
                            "x": "Rezago",
                            "y": "ACF",
                        },
                        title="Autocorrelación",
                    )
                )

            except Exception as e:
                st.info(str(e))

    with tab[2]:
        st.latex(
            r"B_t=\alpha+\rho B_{t-1}"
            r"+\sum_{j=0}^L\beta_j T_{t-j}"
            r"+\sum_{j=0}^L\gamma_j PM_{t-j}+u_t"
        )

        if necesita(DATA, [Y, T, P]):
            c1, c2, c3 = st.columns(3)

            L = c1.slider(
                "Rezagos L", 0, 3, 2
            )

            ar = c2.checkbox(
                "Brecha rezagada", True
            )

            h = c3.slider(
                "Ancho HAC", 0, 8, 3
            )

            try:
                m, o, z, X = ardl(
                    DATA, L, ar, h
                )

                b = bloques(m, L)

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Meses", len(z)
                )

                c2.metric(
                    "R² ajustado",
                    f"{o.rsquared_adj:.4f}",
                )

                c3.metric(
                    "AIC",
                    f"{o.aic:.3f}",
                )

                c4.metric(
                    "BIC",
                    f"{o.bic:.3f}",
                )

                st.subheader(
                    "Coeficientes e intervalos HAC"
                )

                coef = tabla_coef(m)

                st.dataframe(
                    coef,
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader(
                    "Contrastes conjuntos"
                )

                st.dataframe(
                    b,
                    use_container_width=True,
                    hide_index=True,
                )

                fit = z[
                    ["mes", Y]
                ].copy()

                fit["Ajuste ARDL"] = (
                    o.fittedvalues.to_numpy()
                )

                grafico(
                    px.line(
                        fit.assign(
                            mes=fit.mes.astype(str)
                        ),
                        x="mes",
                        y=[Y, "Ajuste ARDL"],
                        title="Ajuste del modelo dentro de muestra",
                    )
                )

                ci = coef[
                    coef.Regresor != "const"
                ].copy()

                grafico(
                    px.scatter(
                        ci,
                        x="Coeficiente",
                        y="Regresor",
                        error_x=(
                            ci["IC 95 % sup"]
                            - ci["Coeficiente"]
                        ),
                        error_x_minus=(
                            ci["Coeficiente"]
                            - ci["IC 95 % inf"]
                        ),
                        title="Coeficientes HAC e IC 95 %",
                    )
                )

                if L == 2 and ar and h == 3:
                    contraste = pd.DataFrame([
                        {
                            "Control": k,
                            "Artículo": v,
                            "Aplicación": w,
                        }
                        for k, v, w in [
                            (
                                "RNDC contemporáneo",
                                ART["RNDC0"],
                                float(m.params["T_L0"]),
                            ),
                            (
                                "p RNDC contemporáneo",
                                ART["p_RNDC0"],
                                float(m.pvalues["T_L0"]),
                            ),
                            (
                                "p RNDC conjunto",
                                ART["p_RNDC_bloque"],
                                float(
                                    b.iloc[0]["p conjunta HAC"]
                                ),
                            ),
                            (
                                "p SIPSA conjunto",
                                ART["p_SIPSA_bloque"],
                                float(
                                    b.iloc[1]["p conjunta HAC"]
                                ),
                            ),
                        ]
                    ])

                    with st.expander(
                        "Auditoría de reproducción respecto al artículo"
                    ):
                        st.dataframe(
                            contraste,
                            hide_index=True,
                        )

                        exportar(
                            contraste,
                            "contraste_articulo.csv",
                        )

                exportar(
                    coef,
                    "coeficientes_ardl.csv",
                )

            except Exception as e:
                st.error(
                    f"No se pudo estimar el ARDL: {e}"
                )

        else:
            st.markdown(
                "**Referencia publicada, no reestimación:**"
            )

            st.dataframe(
                pd.DataFrame([
                    {
                        "N": ART["N"],
                        "Coef RNDC(t)": ART["RNDC0"],
                        "p HAC RNDC(t)": ART["p_RNDC0"],
                        "p RNDC conjunto": ART["p_RNDC_bloque"],
                        "p SIPSA conjunto": ART["p_SIPSA_bloque"],
                    }
                ]),
                hide_index=True,
            )

    with tab[3]:
        if necesita(DATA, [Y, T, P]):
            try:
                _, o, z, X = ardl(DATA)

                diag, res, vif = diagnosticos(
                    o, z, X
                )

                st.dataframe(
                    diag.T,
                    use_container_width=True,
                )

                st.dataframe(
                    vif,
                    use_container_width=True,
                    hide_index=True,
                )

                a, b = st.columns(2)

                with a:
                    grafico(
                        px.scatter(
                            res,
                            x="Ajuste",
                            y="Residuo",
                            title="Residuo frente a ajuste",
                        )
                    )

                with b:
                    grafico(
                        px.histogram(
                            res,
                            x="Residuo",
                            nbins=15,
                            title="Distribución residual",
                        )
                    )

                grafico(
                    px.bar(
                        res.assign(
                            mes=res.mes.astype(str)
                        ),
                        x="mes",
                        y="Cook",
                        title="Distancia de Cook",
                    )
                )

                exportar(
                    res,
                    "diagnosticos_residuos.csv",
                )

            except Exception as e:
                st.info(str(e))

    with tab[4]:
        if necesita(DATA, [Y, T, P]):
            if st.button(
                "Comparar especificaciones",
                type="primary",
            ):
                t = robustez(DATA)

                st.dataframe(
                    t,
                    use_container_width=True,
                    hide_index=True,
                )

                exportar(
                    t,
                    "robustez.csv",
                )

                if "p RNDC/ICTC" in t:
                    vista = t[
                        t["Estado"] == "OK"
                    ].copy()

                    if len(vista):
                        grafico(
                            px.bar(
                                vista,
                                x="Especificación",
                                y="p RNDC/ICTC",
                                title="Sensibilidad del bloque de transporte",
                            )
                        )

            a, b = st.columns(2)

            with a:
                if st.button(
                    "MCO frente a Huber"
                ):
                    try:
                        st.dataframe(
                            huber(DATA),
                            hide_index=True,
                        )
                    except Exception as e:
                        st.error(str(e))

            with b:
                if st.button(
                    "21 ventanas móviles (36 meses)"
                ):
                    try:
                        v = ventanas(DATA)

                        st.dataframe(
                            v,
                            hide_index=True,
                        )

                        grafico(
                            px.line(
                                v,
                                x="Fin",
                                y=[
                                    "Suma RNDC",
                                    "Suma SIPSA",
                                ],
                                title="Estabilidad de sumas puntuales",
                            )
                        )

                        exportar(
                            v,
                            "ventanas.csv",
                        )

                    except Exception as e:
                        st.error(str(e))

    with tab[5]:
        if len(VARS) >= 2:
            elegir = st.multiselect(
                "Series",
                VARS,
                default=VARS[:min(3, len(VARS))],
                format_func=etiqueta,
                key="pca",
            )

            k = st.slider(
                "Número de clusters",
                2, 6, 3,
            )

            if (
                st.button("Ejecutar PCA + K-Means")
                and len(elegir) >= 2
            ):
                try:
                    pts, cargas, var = pca_clusters(
                        DATA, elegir, k
                    )

                    st.metric(
                        "Varianza explicada PC1 + PC2",
                        f"{sum(var):.1%}",
                    )

                    grafico(
                        px.scatter(
                            pts,
                            x="PC1",
                            y="PC2",
                            color="Grupo",
                            hover_data=["Mes"],
                            title="Meses agrupados por patrón",
                        )
                    )

                    st.dataframe(
                        cargas,
                        use_container_width=True,
                    )

                    grafico(
                        px.imshow(
                            cargas,
                            text_auto=".2f",
                            title="Cargas de componentes",
                        )
                    )

                    exportar(
                        pts,
                        "pca_clusters.csv",
                    )

                except Exception as e:
                    st.info(str(e))

    with tab[6]:
        if len(VARS) >= 2:
            elegir = st.multiselect(
                "Variables endógenas",
                VARS,
                default=VARS[:min(3, len(VARS))],
                max_selections=3,
                format_func=etiqueta,
                key="varvars",
            )

            lags = st.slider(
                "Rezagos VAR",
                1, 3, 1,
            )

            if len(elegir) >= 2:
                c1, c2 = st.columns(2)

                with c1:
                    if st.button(
                        "Estimar VAR e IRF"
                    ):
                        try:
                            m, irf = var_exploratorio(
                                DATA,
                                elegir,
                                lags,
                                6,
                            )

                            st.write({
                                "N": m.nobs,
                                "AIC": m.aic,
                                "BIC": m.bic,
                                "Estable": m.is_stable(),
                            })

                            tmp = []

                            for i, respuesta in enumerate(elegir):
                                tmp.extend(
                                    {
                                        "Horizonte": h,
                                        "Respuesta": etiqueta(respuesta),
                                        "Valor": irf[h, i, 0],
                                    }
                                    for h in range(7)
                                )

                            grafico(
                                px.line(
                                    pd.DataFrame(tmp),
                                    x="Horizonte",
                                    y="Valor",
                                    color="Respuesta",
                                    title=(
                                        "Respuestas a innovación en "
                                        f"{etiqueta(elegir[0])}"
                                    ),
                                )
                            )

                        except Exception as e:
                            st.info(str(e))

                with c2:
                    if st.button(
                        "Diagnóstico Johansen / VECM"
                    ):
                        try:
                            test, joh = cointegracion(
                                DATA, elegir
                            )

                            st.dataframe(
                                test,
                                hide_index=True,
                            )

                            if joh is not None:
                                st.dataframe(
                                    joh,
                                    hide_index=True,
                                )
                            else:
                                st.info(
                                    "Los indicadores no justifican VECM "
                                    "bajo este diagnóstico I(1)."
                                )

                        except Exception as e:
                            st.info(str(e))

    with tab[7]:
        if necesita(DATA, [Y, T, P]):
            elegir = st.multiselect(
                "Comparar algoritmos",
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
                key="mlsel",
            )

            if (
                st.button(
                    "Evaluación cronológica",
                    type="primary",
                )
                and elegir
            ):
                try:
                    met, pred, datos, cols, inicio = (
                        ml_cronologico(
                            DATA, elegir
                        )
                    )

                    st.dataframe(
                        met,
                        use_container_width=True,
                        hide_index=True,
                    )

                    grafico(
                        px.line(
                            pred,
                            x="Mes",
                            y="Predicho",
                            color="Modelo",
                            title="Predicción un mes adelante",
                        )
                    )

                    reales = pred.drop_duplicates(
                        "Mes"
                    )[["Mes", "Real"]]

                    grafico(
                        px.line(
                            reales,
                            x="Mes",
                            y="Real",
                            title="Brecha realmente observada",
                        )
                    )

                    exportar(
                        pred,
                        "pronosticos.csv",
                    )

                    modelos = [
                        x for x in elegir
                        if x in [
                            "Ridge",
                            "Lasso",
                            "SVR",
                            "MLP",
                        ]
                    ]

                    if modelos:
                        imp = xai_permutacion(
                            datos,
                            cols,
                            inicio,
                            modelos[0],
                        )

                        st.subheader(
                            f"Explicabilidad: {modelos[0]}"
                        )

                        grafico(
                            px.bar(
                                imp,
                                x="Importancia",
                                y="Regresor",
                                orientation="h",
                                title="Importancia por permutación",
                            )
                        )

                        exportar(
                            imp,
                            "xai.csv",
                        )

                except Exception as e:
                    st.error(str(e))

    with tab[8]:
        if necesita(DATA, [Y, T, P]):
            a, b, c = st.columns(3)

            with a:
                variable = st.selectbox(
                    "Indicador del escenario",
                    [T, P],
                    format_func=etiqueta,
                )

            with b:
                shock = st.number_input(
                    "Variación hipotética (p. p.)",
                    value=1.0,
                    step=0.5,
                )

            with c:
                hor = st.slider(
                    "Horizonte (meses)",
                    1, 12, 6,
                )

            if st.button(
                "Simular",
                type="primary",
            ):
                try:
                    m, _, _, _ = ardl(DATA)

                    t = trayectorias(
                        m,
                        variable,
                        hor,
                        shock,
                    )

                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=t.Mes,
                            y=t["IC sup"],
                            line=dict(width=0),
                            showlegend=False,
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=t.Mes,
                            y=t["IC inf"],
                            fill="tonexty",
                            name="IC 95 %",
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=t.Mes,
                            y=t.Respuesta,
                            mode="lines+markers",
                            name="Respuesta condicionada",
                        )
                    )

                    fig.add_hline(
                        y=0,
                        line_dash="dot",
                    )

                    grafico(fig)

                    st.dataframe(
                        t,
                        hide_index=True,
                    )

                    exportar(
                        t,
                        "trayectoria.csv",
                    )

                except Exception as e:
                    st.error(str(e))

    with tab[9]:
        st.subheader(
            "Episodio de abastecimiento: abril–agosto de 2021"
        )

        datos = DATA[
            DATA["mes"].between(
                pd.Period("2021-04"),
                pd.Period("2021-08"),
            )
        ]

        if VARS:
            st.dataframe(
                datos[
                    ["mes"] + VARS
                ].assign(
                    mes=lambda x: x.mes.astype(str)
                ),
                hide_index=True,
                use_container_width=True,
            )

            serie_grafico(
                datos,
                [VARS[0]],
                "Dinámica durante el episodio",
            )

        if necesita(DATA, [T, P]):
            try:
                sen = semaforo(DATA)

                st.dataframe(
                    sen[
                        sen.Mes.str.startswith("2021")
                    ],
                    hide_index=True,
                )

                grafico(
                    px.scatter(
                        sen,
                        x="RNDC",
                        y="SIPSA",
                        color="Semáforo",
                        hover_data=["Mes"],
                        title="Clasificación histórica por percentiles",
                    )
                )

                exportar(
                    sen,
                    "semaforo_retroactivo.csv",
                )

            except Exception as e:
                st.info(str(e))

        st.markdown(
            "**Dato DANE documentado:** abastecimiento de El Potrerillo: "
            "4.618 t (primera quincena de abril de 2021) y "
            "1.987 t (primera quincena de mayo de 2021)."
        )


elif pagina == "📑 Generador de informes":
    st.title(
        "Generador de informes del observatorio"
    )

    st.caption(
        "Informe construido con las series efectivamente cargadas "
        "y los modelos que puedan estimarse."
    )

    fecha = datetime.now().strftime(
        "%Y-%m-%d %H:%M"
    )

    lineas = [
        "# Informe estadístico Pasto–Colombia",
        f"Fecha: {fecha}",
        "",
        "## Cobertura",
    ]

    for c in [Y, T, P, I]:
        lineas.append(
            f"- {etiqueta(c)}: "
            f"{int(DATA[c].count()) if c in DATA else 0} meses"
        )

    lineas.append(
        "\n## Resultados mensuales disponibles"
    )

    for c in VARS:
        z = DATA[
            ["mes", c]
        ].dropna()

        lineas.append(
            f"- {etiqueta(c)}: "
            f"media {z[c].mean():.4f}; "
            f"mínimo {z[c].min():.4f}; "
            f"máximo {z[c].max():.4f} "
            f"({len(z)} meses)."
        )

    if all(
        c in DISP for c in [Y, T, P]
    ):
        try:
            m, o, z, _ = ardl(DATA)
            b = bloques(m)

            lineas.extend([
                "\n## ARDL L2, HAC(3)",
                (
                    f"N={len(z)}; "
                    f"R² ajustado={o.rsquared_adj:.5f}; "
                    f"AIC={o.aic:.5f}."
                ),
                (
                    f"RNDC contemporáneo={m.params['T_L0']:.6f} "
                    f"(p HAC={m.pvalues['T_L0']:.5f})."
                ),
                (
                    f"p conjunto RNDC="
                    f"{b.iloc[0]['p conjunta HAC']:.5f}; "
                    f"p conjunto SIPSA="
                    f"{b.iloc[1]['p conjunta HAC']:.5f}."
                ),
            ])

        except Exception as e:
            lineas.append(
                f"\nARDL no estimable: {e}"
            )

    else:
        lineas.append(
            "\nLa regresión no se recalcula sin las tres "
            "series auténticas completas."
        )

    lineas.extend([
        "\n## Procedencia",
        (
            "DANE: IPC, SIPSA, ICTC; RNDC: Ministerio de Transporte "
            "(índice construido por autores)."
        ),
        f"https://github.com/{MAIN}",
        f"https://github.com/{SIPSA}",
    ])

    texto = "\n\n".join(lineas)

    st.markdown(texto)

    st.download_button(
        "⬇️ Descargar informe Markdown",
        texto.encode("utf-8-sig"),
        file_name="informe_observatorio.md",
        mime="text/markdown",
    )

    exportar(
        DATA,
        "base_observatorio.csv",
    )


elif pagina == "💬 Asistente estadístico":
    st.title("Asistente de consultas")

    st.caption(
        "Pregunta por meses, promedios, rangos "
        "o las variables del estudio."
    )

    if "dialogo_obs" not in st.session_state:
        st.session_state.dialogo_obs = []

    for item in st.session_state.dialogo_obs:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    pregunta = st.chat_input(
        "Ej.: ¿cuánto fue RNDC en mayo de 2021?"
    )

    if pregunta:
        q = normal(pregunta)

        if any(
            x in q for x in [
                "sipsa",
                "potrerillo",
                "mayorista",
            ]
        ):
            c = P

        elif any(
            x in q for x in [
                "rndc",
                "flete",
            ]
        ):
            c = T

        elif "ictc" in q:
            c = I

        else:
            c = Y

        mes = re.search(
            r"(20(?:18|19|2[0-4]))[- /](0?[1-9]|1[0-2])",
            q,
        )

        meses = {
            "enero": 1,
            "febrero": 2,
            "marzo": 3,
            "abril": 4,
            "mayo": 5,
            "junio": 6,
            "julio": 7,
            "agosto": 8,
            "septiembre": 9,
            "octubre": 10,
            "noviembre": 11,
            "diciembre": 12,
        }

        if mes:
            periodo = pd.Period(
                f"{mes.group(1)}-{int(mes.group(2)):02d}"
            )

        else:
            m2 = re.search(
                r"(enero|febrero|marzo|abril|mayo|junio|julio|"
                r"agosto|septiembre|octubre|noviembre|diciembre)"
                r"\s+(?:de\s+)?(20\d{2})",
                q,
            )

            periodo = (
                pd.Period(
                    f"{m2.group(2)}-{meses[m2.group(1)]:02d}"
                )
                if m2 else None
            )

        if any(
            x in q
            for x in [
                "causa",
                "bloqueo",
                "causal",
            ]
        ):
            res = (
                "Los datos describen asociaciones observacionales. "
                "No identifican impactos causales de bloqueos."
            )

        elif c not in DISP:
            res = (
                f"{etiqueta(c)} todavía no está disponible "
                "en las fuentes recuperadas."
            )

        elif periodo is not None:
            vals = DATA.loc[
                DATA.mes == periodo, c
            ]

            if len(vals) and pd.notna(vals.iloc[0]):
                res = (
                    f"{etiqueta(c)} en {periodo}: "
                    f"**{float(vals.iloc[0]):.5f}**."
                )

            else:
                res = (
                    f"No existe valor observado para "
                    f"{etiqueta(c)} en {periodo}."
                )

        elif any(
            x in q
            for x in [
                "promedio",
                "media",
            ]
        ):
            res = (
                f"Promedio de {etiqueta(c)}: "
                f"**{DATA[c].mean():.5f}**, basado en "
                f"{DATA[c].count()} meses."
            )

        elif any(
            x in q
            for x in [
                "maximo",
                "minimo",
                "pico",
            ]
        ):
            z = DATA[
                ["mes", c]
            ].dropna()

            idx = (
                z[c].idxmin()
                if "minimo" in q
                else z[c].idxmax()
            )

            res = (
                f"{etiqueta(c)}: "
                f"{z.loc[idx, c]:.5f} "
                f"en {z.loc[idx, 'mes']}."
            )

        else:
            res = (
                "Puedo consultar IPC, brecha, RNDC, SIPSA e ICTC. "
                "Indica una variable y un mes (AAAA-MM), "
                "o pide el promedio o máximo."
            )

        st.session_state.dialogo_obs.extend([
            {
                "role": "user",
                "content": pregunta,
            },
            {
                "role": "assistant",
                "content": res,
            },
        ])

        st.rerun()


else:
    st.title(
        "Fuentes, metodología y créditos"
    )

    st.markdown("""
**Universidad de Nariño · Seminario de Investigación Regional II**

**Investigación:** *Costos del transporte de carga, precios mayoristas de alimentos y brecha inflacionaria Pasto–Colombia: evidencia mensual 2018–2024.*

**Autores del artículo:** Juan David Granja Miramag, Helmer Andrés Rivera Vallejos y Yeison Alexander Ortega Rosero.

**Fuentes:** DANE (IPC, ICTC, SIPSA-P y SIPSA-A); Ministerio de Transporte (registros RNDC). GitHub conserva respaldos originales y productos derivados, no sustituye la autoridad estadística de las fuentes primarias.

**Advertencia metodológica.** Todos los modelos son asociativos, no causales. No se imputan ni interpolan observaciones. Se distinguen 84 meses de descripción, 59 de muestra común y 56 para DL/ARDL. La brecha corresponde a la ciudad de Pasto frente al total nacional, no a todo el departamento de Nariño. El índice RNDC fue construido por los autores a partir de registros oficiales; no es un índice publicado por el Ministerio. RNDC e ICTC miden conceptos diferentes. La validación predictiva es seudofuera de muestra y el semáforo es retrospectivo, no una alerta de cierres viales.

**Acerca de este observatorio.** Reproduce la estructura analítica y las restricciones metodológicas del artículo y su notebook de investigación. Los contrastes solo se muestran como reproducidos cuando se recalculan sobre insumos auténticos y pasan las verificaciones.

**Próxima etapa del proyecto:** artículo final y sistema RAG documental; no forman parte de esta aplicación.
""")

    st.markdown(
        f"**Repositorio RNDC:** https://github.com/{MAIN}"
    )

    st.markdown(
        f"**Respaldo SIPSA:** "
        f"https://github.com/{SIPSA}/releases/tag/{TAG}"
    )

    st.markdown(
        f"**Notebook:** "
        f"https://colab.research.google.com/drive/{NB_ID}"
    )

    with st.expander(
        "Auditoría de integridad y errores de fuentes"
    ):
        st.dataframe(
            AUDITORIA,
            hide_index=True,
            use_container_width=True,
        )

        for e in INCIDENCIAS:
            st.write(e)


st.divider()

st.caption(
    "Observatorio Pasto–Nariño · DANE + RNDC · "
    "Investigación económica 2018–2024"
)
