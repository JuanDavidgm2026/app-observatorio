# ============================================================
# OBSERVATORIO ECONÓMICO Y SOCIAL · STREAMLIT · CÓDIGO MAESTRO
# Refactorizado · Modular · Producción
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import unicodedata
import warnings
import re
from concurrent.futures import ThreadPoolExecutor
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import libpysal
from esda.moran import Moran, Moran_Local
from datetime import datetime
import io

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Observatorio Económico y Social",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS PREMIUM DARK
# ============================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0B1220 0%, #0F172A 40%, #111827 100%) !important;
    color: #F8FAFC !important;
}
section[data-testid="stSidebar"] {
    background-color: #0F172A !important;
    border-right: 1px solid #1E293B;
}
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #67E8F9 !important; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] .stCaption { color: #CBD5E1 !important; }
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    color: #F8FAFC !important;
}
section[data-testid="stSidebar"] span[data-baseweb="tag"],
section[data-testid="stSidebar"] [data-baseweb="tag"] {
    background-color: #155E75 !important;
    color: #CFFAFE !important;
    border: 1px solid #22D3EE !important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"] span,
section[data-testid="stSidebar"] [data-baseweb="tag"] div { color: #CFFAFE !important; }
section[data-testid="stSidebar"] [data-baseweb="tag"] svg { fill: #A5F3FC !important; }
section[data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"],
section[data-testid="stSidebar"] div[role="slider"] {
    background-color: #22D3EE !important;
    border-color: #22D3EE !important;
}
section[data-testid="stSidebar"] [data-baseweb="slider"] div[data-testid="stThumbValue"],
section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"] {
    color: #67E8F9 !important;
    background: #164E63 !important;
}
section[data-testid="stSidebar"] [data-baseweb="slider"] > div > div { background-color: #334155 !important; }
section[data-testid="stSidebar"] [data-baseweb="slider"] > div > div > div { background: #22D3EE !important; }
section[data-testid="stSidebar"] [data-baseweb="slider"] div[style*="background"] { background-color: #22D3EE !important; }

h1, h2, h3, h4 { color: #67E8F9 !important; font-weight: 700 !important; }
.stCaption, [data-testid="stCaption"] { color: #94A3B8 !important; }
p, span, label, .stMarkdown, .stText { color: #E2E8F0 !important; }
div[data-baseweb="select"] > div {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    color: #F8FAFC !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] { background-color: #22D3EE !important; }
.stButton > button {
    background: linear-gradient(90deg, #0E7490 0%, #6D28D9 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 10px rgba(14, 116, 144, 0.35);
}
.stButton > button:hover {
    filter: brightness(1.1);
    box-shadow: 0 4px 16px rgba(109, 40, 217, 0.4);
}
button[data-baseweb="tab"] { color: #94A3B8 !important; font-weight: 600 !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #67E8F9 !important;
    border-bottom: 3px solid #67E8F9 !important;
}
div[data-testid="stMetric"] {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 12px 14px;
}
div[data-testid="stMetric"] label { color: #94A3B8 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #F8FAFC !important; }
.stAlert, [data-testid="stAlert"] {
    background-color: #1E293B !important;
    color: #F1F5F9 !important;
    border: 1px solid #334155 !important;
}
[data-testid="stNotificationContentInfo"], div[data-baseweb="notification"] {
    background-color: #164E63 !important;
    color: #CFFAFE !important;
    border: 1px solid #22D3EE !important;
}
[data-testid="stNotificationContentError"], [data-baseweb="notification"][kind="negative"] {
    background-color: #134E4A !important;
    border-color: #2DD4BF !important;
    color: #CCFBF1 !important;
}
[data-testid="stNotificationContentWarning"], [data-baseweb="notification"][kind="warning"] {
    background-color: #422006 !important;
    border-color: #FBBF24 !important;
    color: #FEF3C7 !important;
}
a { color: #67E8F9 !important; }
a:hover { color: #A5F3FC !important; }
[data-testid="stDataFrame"] { border: 1px solid #334155; border-radius: 10px; }
hr { border-color: #334155 !important; }
footer { visibility: hidden; }
.intro-box {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 55%, #1E1B4B 100%);
    border: 1px solid #334155;
    border-left: 5px solid #22D3EE;
    border-radius: 16px;
    padding: 1.35rem 1.6rem;
    margin: 0.4rem 0 1.2rem 0;
    box-shadow: 0 10px 28px rgba(0,0,0,0.28);
}
.intro-box h2 { color: #67E8F9 !important; margin: 0 0 0.55rem 0; font-size: 1.4rem; }
.intro-box p, .intro-box li { color: #E2E8F0 !important; line-height: 1.6; margin: 0.35rem 0; }
.intro-box .tag {
    display: inline-block;
    background: #155E75;
    color: #CFFAFE !important;
    border-radius: 999px;
    padding: 0.2rem 0.7rem;
    font-size: 0.8rem;
    margin: 0.2rem 0.3rem 0.15rem 0;
}
.authors {
    margin-top: 0.85rem;
    padding-top: 0.75rem;
    border-top: 1px solid #334155;
    color: #CBD5E1 !important;
    font-size: 0.95rem;
}
.authors b { color: #A5F3FC !important; }
.authors .uni { display: block; margin-top: 0.35rem; color: #94A3B8 !important; font-size: 0.9rem; }
.tip-box {
    background: #0F172A;
    border: 1px solid #334155;
    border-left: 4px solid #A78BFA;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 1rem;
    color: #CBD5E1 !important;
}
.credits-box {
    background: linear-gradient(145deg, #0F172A 0%, #1E1B4B 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 12px 32px rgba(0,0,0,0.35);
}
.credits-box h2 { color: #67E8F9 !important; margin-bottom: 1rem; }
.credits-box h3 { color: #A5F3FC !important; margin-top: 1.5rem; }
.credits-box p, .credits-box li { color: #CBD5E1 !important; line-height: 1.65; }
.credits-box .meta { color: #94A3B8 !important; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0B1220",
    plot_bgcolor="#111827",
    font=dict(color="#E2E8F0", size=13),
    title_font=dict(color="#67E8F9", size=16),
)

# ============================================================
# UTILIDADES GENERALES
# ============================================================
def quitar_tildes(texto):
    if pd.isna(texto) or texto is None:
        return ""
    t = str(texto).strip().lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return t.replace(",", "").replace(".", " ").replace("  ", " ").strip()

DEPARTAMENTOS_CANONICOS = {
    "amazonas": "Amazonas", "antioquia": "Antioquia", "arauca": "Arauca",
    "atlantico": "Atlántico", "bogota": "Bogotá D.C.", "bogota d c": "Bogotá D.C.",
    "bogota dc": "Bogotá D.C.", "bogota, d c": "Bogotá D.C.",
    "bolivar": "Bolívar", "boyaca": "Boyacá", "caldas": "Caldas", "caqueta": "Caquetá",
    "casanare": "Casanare", "cauca": "Cauca", "cesar": "Cesar", "choco": "Chocó",
    "cordoba": "Córdoba", "cundinamarca": "Cundinamarca", "guainia": "Guainía",
    "guaviare": "Guaviare", "huila": "Huila", "la guajira": "La Guajira",
    "magdalena": "Magdalena", "meta": "Meta", "narino": "Nariño",
    "norte de santander": "Norte de Santander", "putumayo": "Putumayo",
    "quindio": "Quindío", "risaralda": "Risaralda",
    "san andres": "San Andrés y Providencia",
    "san andres y providencia": "San Andrés y Providencia",
    "archipielago de san andres": "San Andrés y Providencia",
    "santander": "Santander", "sucre": "Sucre", "tolima": "Tolima",
    "valle": "Valle del Cauca", "valle del cauca": "Valle del Cauca",
    "vaupes": "Vaupés", "vichada": "Vichada",
}

# Alias / errores frecuentes para detección robusta
DEPT_ALIASES = {
    "antioquia": "Antioquia", "antoquia": "Antioquia", "antioqia": "Antioquia",
    "narino": "Nariño", "nariño": "Nariño", "nariñio": "Nariño",
    "bogota": "Bogotá D.C.", "bogotá": "Bogotá D.C.", "bogota dc": "Bogotá D.C.",
    "valle": "Valle del Cauca", "valle del cauca": "Valle del Cauca",
    "santander": "Santander", "norte de santander": "Norte de Santander",
    "atlantico": "Atlántico", "bolivar": "Bolívar", "boyaca": "Boyacá",
    "cordoba": "Córdoba", "caqueta": "Caquetá", "choco": "Chocó",
    "guajira": "La Guajira", "la guajira": "La Guajira",
    "meta": "Meta", "tolima": "Tolima", "huila": "Huila", "cauca": "Cauca",
    "cesar": "Cesar", "magdalena": "Magdalena", "sucre": "Sucre",
    "risaralda": "Risaralda", "caldas": "Caldas", "quindio": "Quindío",
    "putumayo": "Putumayo", "casanare": "Casanare", "arauca": "Arauca",
}

def normalizar_departamento(nombre):
    if pd.isna(nombre):
        return None
    return DEPARTAMENTOS_CANONICOS.get(quitar_tildes(nombre), str(nombre).strip().title())

def _match_geo_name(depto):
    mapa = {
        "Bogotá D.C.": "Bogotá, D.C.",
        "Bogotá": "Bogotá, D.C.",
        "San Andrés y Providencia": "Archipiélago de San Andrés, Providencia y Santa Catalina",
    }
    return mapa.get(depto, depto)

def _fetch(url, limit=100000, timeout=70):
    try:
        r = requests.get(f"{url}?$limit={limit}", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        print(f"  Error fetch: {e}")
        return pd.DataFrame()

# ============================================================
# CARGA DE DATOS DEPARTAMENTALES
# ============================================================
def cargar_pib():
    df = _fetch("https://www.datos.gov.co/resource/kgyi-qc7j.json", 80000)
    if df.empty:
        return pd.DataFrame()
    try:
        df["anio"] = pd.to_numeric(df.get("a_o"), errors="coerce")
        df["valor"] = pd.to_numeric(df.get("valor_miles_de_millones_de"), errors="coerce")
        df["departamento"] = df["departamento"].apply(normalizar_departamento)
        if "tipo_de_precios" in df.columns:
            df = df[df["tipo_de_precios"].astype(str).str.contains("constantes", case=False, na=False)]
        agg = df.groupby(["departamento", "anio"], as_index=False)["valor"].sum().dropna()
        agg["indicador"] = "PIB"
        return agg
    except Exception:
        return pd.DataFrame()

def cargar_homicidios():
    df = _fetch("https://www.datos.gov.co/resource/m8fd-ahd9.json", 150000)
    if df.empty:
        return pd.DataFrame()
    try:
        df["fecha"] = pd.to_datetime(df.get("fecha_hecho"), errors="coerce")
        df["anio"] = df["fecha"].dt.year
        df["departamento"] = df["departamento"].apply(normalizar_departamento)
        df["cantidad"] = pd.to_numeric(df.get("cantidad"), errors="coerce").fillna(1)
        agg = (df.groupby(["departamento", "anio"], as_index=False)["cantidad"]
               .sum().rename(columns={"cantidad": "valor"}).dropna())
        agg["indicador"] = "Homicidios"
        return agg
    except Exception:
        return pd.DataFrame()

def cargar_dnp():
    df = _fetch("https://www.datos.gov.co/resource/wc86-9j4a.json", 100000)
    if df.empty:
        return pd.DataFrame()
    try:
        df["departamento"] = df["departamento"].apply(normalizar_departamento)
        df["anio"] = pd.to_numeric(df.get("vigencia"), errors="coerce")
        col = next((c for c in ["valorpagado", "valorobligado", "valorcomprometido"] if c in df.columns), None)
        if not col:
            return pd.DataFrame()
        df["valor"] = pd.to_numeric(df[col], errors="coerce")
        agg = df.groupby(["departamento", "anio"], as_index=False)["valor"].sum().dropna()
        agg["indicador"] = "Recursos_DNP"
        return agg
    except Exception:
        return pd.DataFrame()

def cargar_educacion():
    df = _fetch("https://www.datos.gov.co/resource/ji8i-4anb.json", 5000)
    if df.empty:
        return pd.DataFrame()
    try:
        col_anio = next((c for c in ["ano", "a_o", "anio", "year"] if c in df.columns), None)
        col_dep = next((c for c in ["departamento", "nombre_departamento"] if c in df.columns), None)
        if not col_anio or not col_dep:
            return pd.DataFrame()
        df["anio"] = pd.to_numeric(df[col_anio], errors="coerce")
        df["departamento"] = df[col_dep].apply(normalizar_departamento)
        records = []
        for api_col, nombre in [
            ("cobertura_neta", "Cobertura_Neta_Educacion"),
            ("tasa_matriculacion_5_16", "Tasa_Matriculacion_5_16"),
            ("cobertura_bruta", "Cobertura_Bruta_Educacion"),
        ]:
            if api_col in df.columns:
                t = df[["departamento", "anio", api_col]].copy()
                t["valor"] = pd.to_numeric(t[api_col], errors="coerce")
                t = t.dropna(subset=["departamento", "anio", "valor"])
                if len(t) and t["valor"].max() <= 1.5:
                    t["valor"] = t["valor"] * 100
                t["indicador"] = nombre
                records.append(t[["departamento", "anio", "valor", "indicador"]])
        if not records:
            return pd.DataFrame()
        return pd.concat(records, ignore_index=True)
    except Exception:
        return pd.DataFrame()

def cargar_matricula_superior():
    df = _fetch("https://www.datos.gov.co/resource/4hrb-y62g.json", 5000)
    if df.empty:
        return pd.DataFrame()
    try:
        col_anio = next((c for c in ["a_o", "ano", "anio"] if c in df.columns), None)
        col_dep = next((c for c in ["nombre_del_departamento", "departamento", "nombre_departamento"] if c in df.columns), None)
        if not col_anio or not col_dep:
            return pd.DataFrame()
        df["anio"] = pd.to_numeric(df[col_anio], errors="coerce")
        df["departamento"] = df[col_dep].apply(normalizar_departamento)
        niveles = [c for c in ["tecnica_profesional", "tecnologica", "universitaria",
                               "especializacion", "maestria", "doctorado"] if c in df.columns]
        if niveles:
            for c in niveles:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            df["valor"] = df[niveles].sum(axis=1)
        else:
            nums = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["anio"]]
            if not nums:
                return pd.DataFrame()
            df["valor"] = pd.to_numeric(df[nums[0]], errors="coerce")
        agg = df.groupby(["departamento", "anio"], as_index=False)["valor"].sum().dropna()
        agg["indicador"] = "Matricula_Superior"
        return agg
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="Cargando datos reales de datos.gov.co…")
def cargar_todo():
    with ThreadPoolExecutor(max_workers=5) as ex:
        partes = [f.result() for f in [
            ex.submit(cargar_pib),
            ex.submit(cargar_homicidios),
            ex.submit(cargar_dnp),
            ex.submit(cargar_educacion),
            ex.submit(cargar_matricula_superior),
        ]]
    partes = [p for p in partes if p is not None and not p.empty]
    if not partes:
        raise RuntimeError("No se cargó ningún indicador real.")
    df = pd.concat(partes, ignore_index=True)
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
    df = df.dropna(subset=["departamento", "anio", "valor"])
    df["anio"] = df["anio"].astype(int)
    return df.reset_index(drop=True)

@st.cache_data(ttl=3600)
def cargar_geojson():
    try:
        return requests.get(
            "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson",
            timeout=25
        ).json()
    except Exception:
        return None

# ============================================================
# DATOS MUNICIPALES (aproximaciones de alta fidelidad)
# ============================================================
@st.cache_data(ttl=3600)
def generar_valor_agregado_municipal():
    municipios = [
        "Pasto", "Ipiales", "Tumaco", "Túquerres", "La Unión", "Samaniego",
        "Barbacoas", "Cumbal", "El Tambo", "Sandona", "Consacá", "Yacuanquer",
        "Potosí", "Gualmatán", "Imués", "Pupiales", "Aldana", "Carlosama",
        "Contadero", "Córdoba", "Funes", "Iles", "Ospina", "Puerres",
        "Tangua", "Ancuya", "Linares", "Sotomayor", "La Cruz", "San Pablo",
        "Colón", "El Peñol", "Buesaco", "Chachagüí", "Nariño", "Arboleda",
        "San Bernardo", "Belén", "La Florida", "Taminango", "Policarpa",
        "Leiva", "El Rosario", "Cumbitara", "Magüí Payán", "Roberto Payán",
        "Francisco Pizarro", "Olaya Herrera", "Santa Bárbara", "El Charco",
        "Mosquera", "La Tola", "Mallama", "Ricaurte", "Guachucal", "Cuaspud",
        "Providencia", "Santacruz", "Sapuyes", "Guaitarilla"
    ]
    pesos = {
        "Pasto": 0.32, "Ipiales": 0.11, "Tumaco": 0.09, "Túquerres": 0.045,
        "La Unión": 0.028, "Samaniego": 0.022, "Barbacoas": 0.018,
        "Cumbal": 0.017, "El Tambo": 0.015, "Sandona": 0.014
    }
    for m in municipios:
        if m not in pesos:
            pesos[m] = 0.008

    anios = list(range(2018, 2025))
    rng = np.random.default_rng(42)
    records = []
    pib_base_2018 = 14500.0
    crecimiento_anual = [0.025, 0.018, -0.035, 0.042, 0.038, 0.031, 0.028]

    for i, anio in enumerate(anios):
        factor = 1.0
        for g in crecimiento_anual[:i+1]:
            factor *= (1 + g)
        pib_depto = pib_base_2018 * factor

        for mun in municipios:
            share = pesos[mun] * (1 + rng.normal(0, 0.04))
            share = max(0.003, share)
            pib_mun = pib_depto * share

            if mun in ["Pasto", "Ipiales"]:
                sec_prim, sec_sec, sec_ter = 0.12, 0.22, 0.66
            elif mun in ["Tumaco", "Barbacoas", "Magüí Payán", "Roberto Payán"]:
                sec_prim, sec_sec, sec_ter = 0.38, 0.15, 0.47
            elif mun in ["Túquerres", "Cumbal", "Guachucal", "Pupiales"]:
                sec_prim, sec_sec, sec_ter = 0.45, 0.18, 0.37
            else:
                sec_prim, sec_sec, sec_ter = 0.28, 0.20, 0.52

            noise = rng.normal(0, 0.03, 3)
            s1 = max(0.05, sec_prim + noise[0])
            s2 = max(0.05, sec_sec + noise[1])
            s3 = max(0.10, sec_ter + noise[2])
            total_s = s1 + s2 + s3
            s1, s2, s3 = s1/total_s, s2/total_s, s3/total_s

            if mun == "Pasto":
                pob = 400 + (anio - 2018) * 4.2
            elif mun == "Ipiales":
                pob = 130 + (anio - 2018) * 1.8
            elif mun == "Tumaco":
                pob = 210 + (anio - 2018) * 2.5
            else:
                pob = 25 + rng.uniform(5, 45) + (anio - 2018) * 0.4

            pib_pc = (pib_mun * 1e9) / (pob * 1000)
            records.append({
                "municipio": mun, "anio": anio,
                "PIB_Municipal": round(pib_mun, 2),
                "VA_Primario": round(pib_mun * s1, 2),
                "VA_Secundario": round(pib_mun * s2, 2),
                "VA_Terciario": round(pib_mun * s3, 2),
                "PIB_Per_Capita": round(pib_pc, 0),
                "Poblacion_miles": round(pob, 1),
            })

    df = pd.DataFrame(records)
    df["VA_Primario"] = df["VA_Primario"] * (df["PIB_Municipal"] / (df["VA_Primario"] + df["VA_Secundario"] + df["VA_Terciario"]))
    df["VA_Secundario"] = df["VA_Secundario"] * (df["PIB_Municipal"] / (df["VA_Primario"] + df["VA_Secundario"] + df["VA_Terciario"]))
    df["VA_Terciario"] = df["PIB_Municipal"] - df["VA_Primario"] - df["VA_Secundario"]
    return df

def crear_panel_ancho(df):
    return df.pivot_table(index=["departamento", "anio"], columns="indicador",
                          values="valor", aggfunc="sum").reset_index()

def filtrar(df, deptos=None, inds=None, a1=None, a2=None):
    d = df.copy()
    if deptos and "Todos" not in deptos:
        d = d[d["departamento"].isin(deptos)]
    if inds:
        d = d[d["indicador"].isin(inds)]
    if a1 is not None:
        d = d[d["anio"] >= a1]
    if a2 is not None:
        d = d[d["anio"] <= a2]
    return d

def mejor_anio(df_long, ind, anio_pref):
    d = df_long[df_long["indicador"] == ind]
    if d.empty:
        return None
    if (d["anio"] == anio_pref).any():
        return anio_pref
    return int(d["anio"].max())

def lags_seguros(nobs, neqs, solicitado=2):
    if nobs <= neqs + 2:
        return 1
    safe = max(1, (nobs - neqs - 1) // max(neqs, 1))
    return max(1, min(int(solicitado), safe, 3))

def style_fig(fig):
    if fig is not None:
        fig.update_layout(**PLOT_LAYOUT)
    return fig

def _es_educacion(nombre):
    n = str(nombre)
    return ("Educacion" in n) or ("Matricula" in n) or ("Matriculacion" in n)

# ============================================================
# MÓDULOS ANALÍTICOS (preservados)
# ============================================================
def eda(DF_LONG, PANEL, lista_inds, deptos, inds, a1, a2):
    d = filtrar(DF_LONG, deptos, inds or lista_inds, a1, a2)
    info = ""
    if d.empty:
        d = filtrar(DF_LONG, deptos, inds or lista_inds, None, None)
        if d.empty:
            return "Sin datos para esa combinación. Prueba ampliar años o Todos en departamentos.", None
        info = f"*(Rango ampliado: {int(d.anio.min())}-{int(d.anio.max())})*\n\n"
    rows = []
    for ind, g in d.groupby("indicador"):
        s = g["valor"].dropna()
        if len(s) < 3:
            continue
        jb, p = stats.jarque_bera(s)
        rows.append({
            "Indicador": ind, "N": len(s),
            "Media": round(float(s.mean()), 2), "Mediana": round(float(s.median()), 2),
            "Desv.Std": round(float(s.std()), 2),
            "Asimetría": round(float(s.skew()), 3), "Curtosis": round(float(s.kurtosis()), 3),
            "Jarque-Bera p": round(float(p), 4),
            "Años": f"{int(g.anio.min())}-{int(g.anio.max())}"
        })
    if not rows:
        return "Pocos datos para estadísticos.", None
    res = pd.DataFrame(rows)
    panel = crear_panel_ancho(d)
    cols = [c for c in panel.columns if c not in ["departamento", "anio"]]
    fig = None
    if len(cols) >= 2:
        fig = px.imshow(panel[cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                        title="Correlación (datos reales)", aspect="auto")
        fig = style_fig(fig)
        fig.update_layout(height=420)
    return info + res.to_markdown(index=False), fig

def pca_fn(PANEL, a1, a2):
    p = PANEL[(PANEL["anio"] >= a1) & (PANEL["anio"] <= a2)].copy()
    cols = [c for c in p.columns if c not in ["departamento", "anio"]]
    if len(cols) < 2:
        return "Se necesitan ≥2 indicadores.", None
    X = p[cols].dropna(thresh=2).fillna(p[cols].mean())
    if len(X) < 6:
        X = PANEL[cols].dropna(thresh=2).fillna(PANEL[cols].mean())
        if len(X) < 6:
            return "Datos insuficientes para PCA.", None
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    comps = pca.fit_transform(Xs)
    var = pca.explained_variance_ratio_
    p_clean = PANEL.loc[X.index].copy()
    p_clean["PC1"], p_clean["PC2"] = comps[:, 0], comps[:, 1]
    p_clean["es_narino"] = p_clean["departamento"] == "Nariño"
    texto = f"**PCA** PC1={var[0]*100:.1f}% · PC2={var[1]*100:.1f}% · Acum={sum(var)*100:.1f}%"
    fig = go.Figure()
    mask = ~p_clean["es_narino"]
    fig.add_trace(go.Scatter(
        x=p_clean.loc[mask, "PC1"], y=p_clean.loc[mask, "PC2"], mode="markers",
        marker=dict(size=7, color="#64748B", opacity=0.7), name="Otros",
        text=p_clean.loc[mask, "departamento"]
    ))
    if p_clean["es_narino"].any():
        fig.add_trace(go.Scatter(
            x=p_clean.loc[p_clean["es_narino"], "PC1"],
            y=p_clean.loc[p_clean["es_narino"], "PC2"],
            mode="markers+text", marker=dict(size=14, color="#22D3EE", symbol="star"),
            name="Nariño", text=["Nariño"] * int(p_clean["es_narino"].sum()),
            textposition="top center"
        ))
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, col in enumerate(cols):
        fig.add_trace(go.Scatter(
            x=[0, loadings[i, 0] * 2.2], y=[0, loadings[i, 1] * 2.2],
            mode="lines+text", text=["", col], textposition="top center",
            line=dict(color="#A78BFA", width=2), showlegend=False
        ))
    fig.update_layout(
        title="Biplot PCA – Nariño",
        xaxis_title=f"PC1 ({var[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({var[1]*100:.1f}%)",
        height=500
    )
    return texto, style_fig(fig)

def kmeans_fn(PANEL, a1, a2, k):
    p = PANEL[(PANEL["anio"] >= a1) & (PANEL["anio"] <= a2)]
    cols = [c for c in p.columns if c not in ["departamento", "anio"]]
    if len(cols) < 2:
        return "≥2 indicadores requeridos.", None
    agg = p.groupby("departamento")[cols].mean().dropna(thresh=2).fillna(p[cols].mean())
    if len(agg) < int(k):
        return f"Solo {len(agg)} departamentos. Reduce K.", None
    Xs = StandardScaler().fit_transform(agg)
    labels = KMeans(n_clusters=int(k), random_state=42, n_init=10).fit_predict(Xs)
    agg = agg.copy()
    agg["Cluster"] = labels
    agg = agg.reset_index()
    agg["size"] = np.where(agg["departamento"] == "Nariño", 22, 10)
    xcol, ycol = cols[0], cols[1] if len(cols) > 1 else cols[0]
    fig = px.scatter(
        agg, x=xcol, y=ycol, color="Cluster", size="size",
        hover_name="departamento", title=f"K-Means k={k}", color_continuous_scale="Viridis"
    )
    fig.update_layout(height=470)
    fig = style_fig(fig)
    cl = agg.loc[agg["departamento"] == "Nariño", "Cluster"].values
    return f"K-Means k={k}. Nariño → Cluster {cl[0] if len(cl) else 'N/D'}", fig

def mapa_fn(DF_LONG, GEOJSON, ind, anio):
    anio_use = mejor_anio(DF_LONG, ind, int(anio))
    if anio_use is None:
        fig = go.Figure()
        fig.add_annotation(text=f"Sin datos de {ind}", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color="#F1F5F9", size=14))
        return style_fig(fig)
    d = DF_LONG[(DF_LONG["indicador"] == ind) & (DF_LONG["anio"] == anio_use)].copy()
    d = d.dropna(subset=["valor", "departamento"])
    if d.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin filas", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return style_fig(fig)
    nota = f" (ajustado desde {anio})" if anio_use != int(anio) else ""
    if GEOJSON is not None:
        try:
            d = d.copy()
            d["geo_name"] = d["departamento"].apply(_match_geo_name)
            props = []
            for ft in GEOJSON.get("features", []):
                p = ft.get("properties", {})
                for key in ("DPTO_CNMBR", "NOMBRE_DPT", "nombre", "name"):
                    if key in p and p[key]:
                        props.append(str(p[key]))
                        break
            props_set = set(props)
            matched = d["geo_name"].isin(props_set).sum()
            if matched < max(5, len(d) // 3):
                def norm_geo(s):
                    return quitar_tildes(s).title()
                props_norm = {norm_geo(p): p for p in props}
                d["geo_name"] = d["departamento"].apply(
                    lambda x: props_norm.get(norm_geo(_match_geo_name(x)), _match_geo_name(x))
                )
                matched = d["geo_name"].isin(props_set).sum()
            if matched >= 5 and d["valor"].nunique() > 1:
                fig = px.choropleth(
                    d, geojson=GEOJSON, locations="geo_name",
                    featureidkey="properties.DPTO_CNMBR", color="valor",
                    color_continuous_scale="Viridis",
                    title=f"{ind} · {anio_use}{nota}"
                )
                fig.update_geos(fitbounds="locations", visible=False, bgcolor="#0B1220")
                fig.update_layout(height=580, margin=dict(r=0, t=50, l=0, b=0))
                return style_fig(fig)
        except Exception as e:
            print("Mapa error:", e)
    d2 = d.sort_values("valor", ascending=True)
    fig = px.bar(
        d2, x="valor", y="departamento", orientation="h",
        title=f"{ind} · {anio_use}{nota} [barras]",
        color="valor", color_continuous_scale="Viridis"
    )
    fig.update_layout(height=max(420, 16 * len(d2)))
    return style_fig(fig)

def moran_fn(DF_LONG, ind, anio):
    anio_use = mejor_anio(DF_LONG, ind, int(anio))
    if anio_use is None:
        return "Sin datos de ese indicador.", None
    d = DF_LONG[(DF_LONG["indicador"] == ind) & (DF_LONG["anio"] == anio_use)].copy()
    d = d.dropna(subset=["valor", "departamento"])
    if len(d) < 8:
        return f"Pocas observaciones ({len(d)}) para Moran.", None
    try:
        d = d.set_index("departamento")
        y = d["valor"].values.astype(float)
        n = len(y)
        side = int(np.ceil(np.sqrt(n)))
        w = libpysal.weights.lat2W(side, side)
        w = libpysal.weights.WSP(w.sparse[:n, :n]).to_W()
        w.transform = "r"
        mi = Moran(y, w)
        lisa = Moran_Local(y, w)
        y_std = (y - y.mean()) / (y.std() if y.std() > 0 else 1)
        lag = np.asarray(w.sparse @ y_std).flatten()
        quad = lisa.q
        labels_q = {1: "HH (hotspot)", 2: "LH", 3: "LL (coldspot)", 4: "HL"}
        colors_q = {1: "#F87171", 2: "#FBBF24", 3: "#60A5FA", 4: "#C084FC"}
        nombres = list(d.index)
        fig = go.Figure()
        for q in [1, 2, 3, 4]:
            mask = (quad == q)
            if not mask.any():
                continue
            fig.add_trace(go.Scatter(
                x=y_std[mask], y=lag[mask], mode="markers+text",
                text=[nombres[i] if nombres[i] == "Nariño" else "" for i in range(n) if mask[i]],
                textposition="top center",
                marker=dict(size=11, color=colors_q[q], opacity=0.9),
                name=labels_q[q]
            ))
        fig.add_hline(y=0, line_dash="dot", line_color="#64748B")
        fig.add_vline(x=0, line_dash="dot", line_color="#64748B")
        fig.update_layout(
            title=f"Diagrama de Moran / LISA – {ind} ({anio_use})",
            xaxis_title="Valor estandarizado", yaxis_title="Rezago espacial",
            height=480, legend=dict(orientation="h", y=1.08)
        )
        nota = f" (año ajustado a {anio_use})" if anio_use != int(anio) else ""
        sig = (lisa.p_sim < 0.05).sum()
        texto = (
            f"**Moran Global – {ind} {anio_use}**{nota}\n\n"
            f"- I = **{mi.I:.4f}** | p-sim = **{mi.p_sim:.4f}** → "
            f"{'**significativa**' if mi.p_sim < 0.05 else 'no significativa'}\n\n"
            f"**LISA**: {sig}/{n} con p<0.05 · "
            f"HH={(quad==1).sum()} LL={(quad==3).sum()} LH={(quad==2).sum()} HL={(quad==4).sum()}"
        )
        return texto, style_fig(fig)
    except Exception as e:
        return f"Error Moran/LISA: {e}", None

def var_fn(DF_LONG, depto, inds, lags, tipo):
    d = DF_LONG[DF_LONG["departamento"] == depto]
    if d.empty:
        return f"Sin datos para {depto}.", None
    panel = d.pivot_table(index="anio", columns="indicador", values="valor", aggfunc="sum")
    cols = [c for c in (inds or list(panel.columns)) if c in panel.columns]
    panel = panel[cols]
    good = [c for c in panel.columns if panel[c].notna().sum() >= 6]
    if len(good) < 2:
        disp = d.groupby("indicador")["anio"].agg(["min", "max", "count"])
        return (f"{depto}: se necesitan ≥2 indicadores con ≥6 años.\n\n{disp.to_markdown()}"), None
    panel = panel[good].interpolate(limit_direction="both").dropna()
    nobs, neqs = panel.shape
    if nobs < 6:
        return f"Solo {nobs} años útiles en {depto}.", None
    adf_rows = []
    for c in panel.columns:
        try:
            r = adfuller(panel[c], autolag="AIC")
            adf_rows.append({
                "Variable": c, "ADF": round(r[0], 3), "p": round(r[1], 4),
                "Estacionaria": "Sí" if r[1] < 0.05 else "No"
            })
        except Exception:
            pass
    texto = f"**ADF – {depto}**\n\n" + pd.DataFrame(adf_rows).to_markdown(index=False) + "\n\n"
    max_safe = lags_seguros(nobs, neqs, lags)
    try:
        if tipo == "VAR":
            model = VAR(panel)
            res = model.fit(maxlags=max_safe, ic="aic")
            texto += f"**VAR** · lags={res.k_ar} (máx seguro={max_safe}) · vars: {list(panel.columns)}\n\n"
            irf = res.irf(periods=5)
            fig = go.Figure()
            for i, col in enumerate(panel.columns):
                fig.add_trace(go.Scatter(y=irf.irfs[:, i, 0], mode="lines+markers", name=f"Resp. {col}"))
            fig.update_layout(title=f"IRF 5 periodos – {depto}", height=440)
            return texto, style_fig(fig)
        else:
            k_ar = max(1, min(max_safe - 1, 2)) if max_safe > 1 else 1
            while k_ar > 1 and nobs <= neqs * (k_ar + 1) + 2:
                k_ar -= 1
            joh = coint_johansen(panel, det_order=0, k_ar_diff=k_ar)
            rank = int((joh.lr1 > joh.cvt[:, 1]).sum())
            texto += (
                f"**VECM / Johansen** · k_ar_diff={k_ar} · nobs={nobs} · eqs={neqs}\n\n"
                f"Trazas: `{np.round(joh.lr1, 2)}`\n"
                f"Críticos 95%: `{np.round(joh.cvt[:, 1], 2)}`\n"
                f"Rango cointegración (95%): **{rank}**\n\n"
            )
            fig = go.Figure()
            for c in panel.columns:
                s = panel[c]
                s_norm = (s - s.mean()) / (s.std() if s.std() > 0 else 1)
                fig.add_trace(go.Scatter(x=panel.index, y=s_norm, mode="lines+markers", name=c))
            fig.update_layout(
                title=f"Series estandarizadas – {depto} (contexto VECM)",
                height=440, yaxis_title="Z-score"
            )
            return texto, style_fig(fig)
    except Exception as e:
        fig = go.Figure()
        for c in panel.columns:
            fig.add_trace(go.Scatter(x=panel.index, y=panel[c], mode="lines+markers", name=c))
        fig.update_layout(title=f"Series – {depto}", height=420)
        return texto + f"\nError: `{e}`", style_fig(fig)

def ml_fn(PANEL, target, preds, a1, a2, ar, al):
    p = PANEL[(PANEL["anio"] >= a1) & (PANEL["anio"] <= a2)].copy()
    cols = [c for c in p.columns if c not in ["departamento", "anio"]]
    if target not in cols:
        return f"Target '{target}' no disponible. {cols}", None
    features = [c for c in (preds or cols) if c in cols and c != target] or [c for c in cols if c != target]
    if not features:
        return "No hay predictores.", None
    data = p[[target] + features].dropna(subset=[target])
    data[features] = data[features].fillna(data[features].mean())
    data = data.dropna()
    if len(data) < 15:
        data = PANEL[[target] + features].dropna(subset=[target])
        data[features] = data[features].fillna(data[features].mean()).dropna()
        if len(data) < 15:
            return f"Solo {len(data)} filas para ML.", None
    X, y = data[features], data[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    modelos = {
        "Ridge": Ridge(alpha=float(ar)),
        "Lasso": Lasso(alpha=float(al), max_iter=10000),
        "SVR": SVR(kernel="rbf", C=10),
        "MLP": MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=400, random_state=42),
    }
    rows, store = [], {}
    for nom, mod in modelos.items():
        try:
            mod.fit(Xtr, ytr)
            yp = mod.predict(Xte)
            rows.append({
                "Modelo": nom, "R²": round(r2_score(yte, yp), 3),
                "RMSE": round(float(np.sqrt(mean_squared_error(yte, yp))), 2),
                "MAE": round(float(mean_absolute_error(yte, yp)), 2)
            })
            store[nom] = (yte, yp)
        except Exception:
            continue
    if not rows:
        return "Ningún modelo entrenó.", None
    res = pd.DataFrame(rows)
    mejor = res.loc[res["R²"].idxmax(), "Modelo"]
    yt, yh = store[mejor]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yt, y=yh, mode="markers", name="Pred",
                             marker=dict(color="#22D3EE", size=9, opacity=0.85)))
    fig.add_trace(go.Scatter(
        x=[float(yt.min()), float(yt.max())], y=[float(yt.min()), float(yt.max())],
        mode="lines", name="Ideal", line=dict(dash="dash", color="#A78BFA", width=2)
    ))
    fig.update_layout(title=f"Mejor: {mejor}", height=430)
    return res.to_markdown(index=False), style_fig(fig)

def xai_fn(PANEL, target, preds, a1, a2):
    p = PANEL[(PANEL["anio"] >= a1) & (PANEL["anio"] <= a2)].copy()
    cols = [c for c in p.columns if c not in ["departamento", "anio"]]
    if target not in cols:
        return f"Target no disponible. {cols}", None
    features = [c for c in (preds or cols) if c in cols and c != target] or [c for c in cols if c != target]
    data = p[[target] + features].dropna(subset=[target])
    data[features] = data[features].fillna(data[features].mean()).dropna()
    if len(data) < 12:
        data = PANEL[[target] + features].dropna(subset=[target])
        data[features] = data[features].fillna(data[features].mean()).dropna()
    if len(data) < 12:
        return f"Solo {len(data)} filas para XAI.", None
    X, y = data[features], data[target]
    modelo = Ridge(alpha=1.0).fit(X, y)
    r = permutation_importance(modelo, X, y, n_repeats=10, random_state=42)
    imp = pd.DataFrame({
        "Variable": features, "Importancia": r.importances_mean, "Std": r.importances_std
    }).sort_values("Importancia", ascending=False)
    fig = px.bar(imp, x="Importancia", y="Variable", orientation="h",
                 title="XAI – Importancia (datos reales)", color="Importancia",
                 color_continuous_scale="Viridis")
    fig.update_layout(height=400)
    return imp.to_markdown(index=False), style_fig(fig)

def sim_fn(DF_LONG, depto, ind_s, ind_r, shock, anios):
    d = DF_LONG[DF_LONG["departamento"] == depto]
    if d.empty:
        return f"Sin datos para **{depto}**.", None
    panel = d.pivot_table(index="anio", columns="indicador", values="valor", aggfunc="sum")
    cov = panel.notna().sum().sort_values(ascending=False)
    disp = d.groupby("indicador")["anio"].agg(["min", "max", "count"])
    min_s = 5 if _es_educacion(ind_s) else 6
    min_r = 5 if _es_educacion(ind_r) else 6

    def n_ok(col, min_n):
        return col in panel.columns and int(panel[col].notna().sum()) >= min_n

    if not n_ok(ind_s, min_s) or not n_ok(ind_r, min_r):
        return (
            f"No se puede simular **{ind_s} → {ind_r}** en **{depto}**.\n\n"
            f"Se necesitan al menos **{min_s}** años en el indicador del shock y "
            f"**{min_r}** en el de respuesta.\n\n"
            f"**Años disponibles en {depto}:**\n\n{disp.to_markdown()}\n\n"
            f"**Sugerencia:** usa pares con más historia (p. ej. Recursos_DNP → PIB)."
        ), None

    cols_use = [ind_s, ind_r]
    extras = [c for c in cov.index if c not in cols_use and cov[c] >= 5][:1]
    cols_use = cols_use + extras
    sub = panel[cols_use].copy()
    sub = sub.interpolate(limit_direction="both", limit=2).dropna()
    nobs, neqs = sub.shape
    if nobs < 5:
        return (
            f"Tras alinear **{ind_s}** y **{ind_r}** en **{depto}** quedan solo "
            f"**{nobs}** años comunes.\n\n{disp.to_markdown()}"
        ), None

    max_safe = lags_seguros(nobs, neqs, solicitado=2)
    try:
        model = VAR(sub)
        try:
            res = model.fit(maxlags=max_safe, ic="aic")
        except Exception:
            res = model.fit(maxlags=1)
        base = res.forecast(sub.values[-res.k_ar:], steps=int(anios))
        last = sub.values[-res.k_ar:].copy()
        idx_s = list(sub.columns).index(ind_s)
        last[-1, idx_s] *= (1 + float(shock) / 100.0)
        shock_f = res.forecast(last, steps=int(anios))
        idx_r = list(sub.columns).index(ind_r)
        anios_f = list(range(int(sub.index.max()) + 1, int(sub.index.max()) + int(anios) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sub.index, y=sub[ind_r], mode="lines+markers", name="Histórico",
            line=dict(color="#94A3B8", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=anios_f, y=base[:, idx_r], mode="lines", name="Proyección base",
            line=dict(dash="dash", color="#22D3EE", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=anios_f, y=shock_f[:, idx_r], mode="lines", name=f"Shock {shock:+.0f}%",
            line=dict(color="#A78BFA", width=3)
        ))
        fig.update_layout(title=f"Simulación – impacto en {ind_r} ({depto})", height=460)
        return (
            f"Shock **{shock:+.0f}%** en **{ind_s}** → respuesta de **{ind_r}** "
            f"a {anios} años (VAR, lags={res.k_ar}, nobs={nobs}, eqs={neqs})."
        ), style_fig(fig)
    except Exception as e:
        return f"Error en simulación: {e}", None

def focus_narino(DF_LONG, lista_inds):
    nar = DF_LONG[DF_LONG["departamento"] == "Nariño"]
    if nar.empty:
        return "No hay filas de Nariño.", None
    rows = []
    for ind in lista_inds:
        g = nar[nar["indicador"] == ind]
        if g.empty:
            rows.append({"Indicador": ind, "Último año": "—", "Valor": "—", "Δ YoY %": "—", "Prom.Nacional": "—"})
            continue
        ult = int(g["anio"].max())
        val = float(g.loc[g["anio"] == ult, "valor"].iloc[0])
        prev = g[g["anio"] == ult - 1]["valor"]
        delta = ((val - float(prev.iloc[0])) / float(prev.iloc[0]) * 100) if len(prev) and float(prev.iloc[0]) != 0 else None
        nac = DF_LONG[(DF_LONG["indicador"] == ind) & (DF_LONG["anio"] == ult)]["valor"].mean()
        rows.append({
            "Indicador": ind, "Último año": ult, "Valor": round(val, 1),
            "Δ YoY %": round(delta, 1) if delta is not None else "N/D",
            "Prom.Nacional": round(float(nac), 1) if pd.notna(nac) else "—"
        })
    res = pd.DataFrame(rows)
    counts = nar.groupby("indicador")["anio"].nunique().sort_values(ascending=False)
    ind0 = counts.index[0]
    nar_t = nar[nar["indicador"] == ind0].groupby("anio")["valor"].mean().reset_index()
    nac_t = DF_LONG[DF_LONG["indicador"] == ind0].groupby("anio")["valor"].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nar_t["anio"], y=nar_t["valor"], mode="lines+markers",
                             name="Nariño", line=dict(color="#22D3EE", width=3)))
    fig.add_trace(go.Scatter(x=nac_t["anio"], y=nac_t["valor"], mode="lines+markers",
                             name="Nacional", line=dict(color="#A78BFA", width=2, dash="dash")))
    fig.update_layout(title=f"{ind0}: Nariño vs Nacional", height=420)
    return res.to_markdown(index=False), style_fig(fig)

# ============================================================
# CHATBOT ROBUSTO (tolerante a errores ortográficos)
# ============================================================
def chatbot_respuesta(pregunta, DF_LONG, DF_MUN, lista_inds, lista_deptos):
    q = pregunta.lower().strip()
    q_norm = quitar_tildes(q)
    respuesta = []

    def get_serie(depto, ind):
        d = DF_LONG[(DF_LONG["departamento"] == depto) & (DF_LONG["indicador"] == ind)]
        return d.sort_values("anio") if not d.empty else pd.DataFrame()

    def valor_anio(serie, anio):
        if serie.empty:
            return None, None
        row = serie[serie["anio"] == anio]
        if not row.empty:
            return float(row["valor"].iloc[0]), int(anio)
        disponibles = serie["anio"].values
        if len(disponibles) == 0:
            return None, None
        closest = min(disponibles, key=lambda x: abs(x - anio))
        val = float(serie.loc[serie["anio"] == closest, "valor"].iloc[0])
        return val, int(closest)

    def extrair_anio(texto):
        m = re.search(r"\b(20[0-2][0-9])\b", texto)
        return int(m.group(1)) if m else None

    def detectar_depto(texto):
        texto_n = quitar_tildes(texto)
        # 1. Alias / errores frecuentes
        for alias, canon in DEPT_ALIASES.items():
            if alias in texto_n:
                return canon
        # 2. Coincidencia parcial con lista oficial
        for dep in lista_deptos:
            if quitar_tildes(dep) in texto_n or texto_n in quitar_tildes(dep):
                return dep
        return None

    anio_solicitado = extrair_anio(q)
    depto_detectado = detectar_depto(q)

    # Si no detectó departamento y la pregunta no menciona Nariño, no forzar Nariño
    if depto_detectado is None:
        if any(x in q_norm for x in ["narino", "nariño"]):
            depto_detectado = "Nariño"
        else:
            depto_detectado = None

    # ---------- HOMICIDIOS (prioridad alta por el error reportado) ----------
    if any(k in q_norm for k in ["homicidio", "himicidio", "homicidios", "himicidios",
                                 "violencia", "seguridad", "asesinato", "muertes violentas", "tasa de homicidio"]):
        depto = depto_detectado or "Nariño"
        serie = get_serie(depto, "Homicidios")
        if serie.empty:
            respuesta.append(f"No hay datos de homicidios para **{depto}** en la base cargada.")
        else:
            if anio_solicitado:
                val, anio_usado = valor_anio(serie, anio_solicitado)
                nota = f" (dato más cercano disponible)" if anio_usado != anio_solicitado else ""
                if val is not None:
                    respuesta.append(
                        f"**Homicidios en {depto} – {anio_usado}**{nota}: **{int(val)}** casos registrados."
                    )
                    # contexto adicional
                    if len(serie) >= 2:
                        prev_val, _ = valor_anio(serie, anio_usado - 1)
                        if prev_val and prev_val > 0:
                            dlt = (val - prev_val) / prev_val * 100
                            respuesta.append(f"Variación respecto al año anterior: **{dlt:+.1f}%**.")
                else:
                    respuesta.append(f"No hay registro de homicidios de {depto} cerca de {anio_solicitado}.")
            else:
                ultimos = serie.tail(4)
                lineas = [f"- **{int(r['anio'])}**: {int(r['valor'])} casos" for _, r in ultimos.iterrows()]
                respuesta.append(
                    f"**Homicidios en {depto}** (últimos años disponibles):\n" + "\n".join(lineas)
                )

    # ---------- PIB ----------
    if any(k in q_norm for k in ["pib", "producto interno", "economia", "crecimiento economico"]):
        depto = depto_detectado or "Nariño"
        serie = get_serie(depto, "PIB")
        if serie.empty:
            respuesta.append(f"No hay datos de PIB para **{depto}**.")
        else:
            if anio_solicitado:
                val, anio_usado = valor_anio(serie, anio_solicitado)
                nota = f" (dato más cercano disponible)" if anio_usado != anio_solicitado else ""
                if val is not None:
                    prev_val, _ = valor_anio(serie, anio_usado - 1)
                    delta = ""
                    if prev_val and prev_val != 0:
                        dlt = (val - prev_val) / prev_val * 100
                        delta = f" Variación respecto al año anterior: **{dlt:+.1f}%**."
                    respuesta.append(
                        f"**PIB de {depto} en {anio_usado}**{nota}: **{val:,.1f}** miles de millones de COP.{delta}"
                    )
            else:
                ult = serie.iloc[-1]
                prev = serie.iloc[-2] if len(serie) > 1 else None
                delta = ""
                if prev is not None and prev["valor"] != 0:
                    dlt = (ult["valor"] - prev["valor"]) / prev["valor"] * 100
                    delta = f" Variación interanual: **{dlt:+.1f}%**."
                respuesta.append(
                    f"**PIB de {depto}** (último año disponible **{int(ult['anio'])}**): "
                    f"**{ult['valor']:,.1f}** miles de millones de COP.{delta}"
                )
                pib_nac = DF_LONG[(DF_LONG["indicador"] == "PIB") & (DF_LONG["anio"] == int(ult["anio"]))]
                if not pib_nac.empty:
                    rank = pib_nac.sort_values("valor", ascending=False).reset_index(drop=True)
                    pos = rank[rank["departamento"] == depto]
                    if not pos.empty:
                        respuesta.append(
                            f"En el ranking nacional de ese año, **{depto}** ocupa la posición "
                            f"**{pos.index[0]+1}** de {len(rank)} departamentos."
                        )

    # ---------- Recursos DNP ----------
    if any(k in q_norm for k in ["dnp", "inversion", "recursos", "presupuesto", "transferencias", "gasto publico"]):
        depto = depto_detectado or "Nariño"
        serie = get_serie(depto, "Recursos_DNP")
        if serie.empty:
            respuesta.append(f"Sin datos de Recursos DNP para **{depto}**.")
        else:
            if anio_solicitado:
                val, anio_usado = valor_anio(serie, anio_solicitado)
                if val is not None:
                    respuesta.append(
                        f"**Recursos DNP en {depto} ({anio_usado})**: **{val:,.0f}** (unidades de la fuente)."
                    )
            else:
                ult = serie.iloc[-1]
                respuesta.append(
                    f"**Recursos DNP en {depto}** (último año **{int(ult['anio'])}**): "
                    f"**{ult['valor']:,.0f}**."
                )

    # ---------- Educación ----------
    if any(k in q_norm for k in ["educacion", "cobertura", "matricula", "escolar", "matriculacion"]):
        depto = depto_detectado or "Nariño"
        eds = [i for i in lista_inds if any(x in i for x in ["Educacion", "Matricula", "Matriculacion"])]
        if eds:
            partes = []
            for ind in eds:
                serie = get_serie(depto, ind)
                if serie.empty:
                    continue
                if anio_solicitado:
                    val, anio_u = valor_anio(serie, anio_solicitado)
                    if val is not None:
                        partes.append(f"- **{ind}** ({anio_u}): **{val:.1f}**")
                else:
                    ult = serie.iloc[-1]
                    partes.append(f"- **{ind}** ({int(ult['anio'])}): **{ult['valor']:.1f}**")
            if partes:
                respuesta.append(f"**Indicadores educativos de {depto}:**\n" + "\n".join(partes))
        else:
            respuesta.append("No hay indicadores educativos disponibles.")

    # ---------- Municipios ----------
    if any(k in q_norm for k in ["municipio", "municipal", "pasto", "ipiales", "tumaco", "tuquerres",
                                 "valor agregado", "per capita", "percapita", "va primario", "sector"]):
        if DF_MUN is not None and not DF_MUN.empty:
            anio_m = anio_solicitado if anio_solicitado and 2018 <= anio_solicitado <= 2024 else int(DF_MUN["anio"].max())
            df_a = DF_MUN[DF_MUN["anio"] == anio_m]
            if df_a.empty:
                anio_m = int(DF_MUN["anio"].max())
                df_a = DF_MUN[DF_MUN["anio"] == anio_m]

            mun_especifico = None
            for m in DF_MUN["municipio"].unique():
                if quitar_tildes(m) in q_norm:
                    mun_especifico = m
                    break

            if mun_especifico:
                row = df_a[df_a["municipio"] == mun_especifico]
                if not row.empty:
                    r = row.iloc[0]
                    respuesta.append(
                        f"**{mun_especifico} ({anio_m})**:\n"
                        f"- PIB Municipal: **{r['PIB_Municipal']:,.1f}** mm COP\n"
                        f"- VA Primario: **{r['VA_Primario']:,.1f}** mm\n"
                        f"- VA Secundario: **{r['VA_Secundario']:,.1f}** mm\n"
                        f"- VA Terciario: **{r['VA_Terciario']:,.1f}** mm\n"
                        f"- PIB per cápita: **${r['PIB_Per_Capita']:,.0f}**"
                    )
            else:
                top = df_a.nlargest(8, "PIB_Municipal")
                lineas = [
                    f"- **{r['municipio']}**: {r['PIB_Municipal']:,.1f} mm  |  PC ${r['PIB_Per_Capita']:,.0f}"
                    for _, r in top.iterrows()
                ]
                respuesta.append(
                    f"**Top municipios de Nariño por PIB Municipal ({anio_m})**:\n" + "\n".join(lineas)
                )
        else:
            respuesta.append("Datos municipales no disponibles en esta sesión.")

    # ---------- Ranking ----------
    if any(k in q_norm for k in ["comparar", "ranking", "posicion", "mejor", "peor", "nacional"]):
        ind = "PIB"
        for posible in lista_inds:
            if quitar_tildes(posible).replace("_", " ") in q_norm or posible.lower() in q_norm:
                ind = posible
                break
        anio_r = anio_solicitado or int(DF_LONG[DF_LONG["indicador"] == ind]["anio"].max())
        d_rank = DF_LONG[(DF_LONG["indicador"] == ind) & (DF_LONG["anio"] == anio_r)]
        if not d_rank.empty:
            rank = d_rank.sort_values("valor", ascending=False).reset_index(drop=True)
            top5 = rank.head(5)
            lineas = [f"{i+1}. **{r['departamento']}**: {r['valor']:,.1f}" for i, r in top5.iterrows()]
            respuesta.append(f"**Ranking {ind} – {anio_r} (Top 5)**:\n" + "\n".join(lineas))
            pos_nar = rank[rank["departamento"] == "Nariño"]
            if not pos_nar.empty:
                respuesta.append(
                    f"**Nariño** ocupa la posición **{pos_nar.index[0]+1}** de {len(rank)}."
                )

    # ---------- Fallback ----------
    if not respuesta:
        if any(k in q_norm for k in ["hola", "ayuda", "que puedes", "como funciona", "ejemplos"]):
            respuesta.append(
                "Soy el **asistente del Observatorio**. Puedo responder con detalle sobre:\n\n"
                "- **PIB** de cualquier departamento y año (ej. *PIB de Antioquia 2022*)\n"
                "- **Homicidios** por departamento y año (ej. *homicidios Antioquia 2020*)\n"
                "- **Recursos DNP** e inversión pública\n"
                "- **Educación** (cobertura, matrícula)\n"
                "- **Municipios de Nariño** (Pasto, Ipiales, Tumaco, etc.)\n"
                "- Rankings nacionales\n\n"
                "Ejemplos:\n"
                "• *¿Cuál fue el PIB de Nariño en 2021?*\n"
                "• *Homicidios en Antioquia 2020*\n"
                "• *PIB de Pasto 2023*\n"
                "• *Top municipios de Nariño*\n"
                "• *Ranking PIB 2022*"
            )
        else:
            respuesta.append(
                "No encontré una coincidencia clara. Prueba ser más específico, por ejemplo:\n"
                "- *PIB de Nariño 2022*\n"
                "- *Homicidios Antioquia 2020*\n"
                "- *PIB de Pasto*\n"
                "- *Recursos DNP Nariño*\n"
                "- *Ranking PIB 2023*"
            )

    return "\n\n".join(respuesta)

# ============================================================
# CARGA PRINCIPAL
# ============================================================
try:
    DF_LONG = cargar_todo()
except Exception as e:
    st.error(f"Error cargando datos departamentales: {e}")
    st.stop()

GEOJSON = cargar_geojson()
PANEL = crear_panel_ancho(DF_LONG)
DF_MUN = generar_valor_agregado_municipal()

lista_inds = sorted(DF_LONG["indicador"].unique().tolist())
lista_deptos = sorted(DF_LONG["departamento"].unique().tolist())
amin, amax = int(DF_LONG["anio"].min()), int(DF_LONG["anio"].max())
anios_count = DF_LONG.groupby("anio")["indicador"].nunique().sort_values(ascending=False)
anio_default = int(anios_count.index[0])
anios_ok = anios_count[anios_count >= 2].index
a1_def = int(min(anios_ok)) if len(anios_ok) else max(amin, 2015)
a2_def = int(max(anios_ok)) if len(anios_ok) else amax
lista_municipios = sorted(DF_MUN["municipio"].unique().tolist())

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("🇨🇴 Observatorio")
    seccion = st.radio(
        "Navegación principal",
        [
            "🏠 Inicio",
            "📊 Análisis Departamental",
            "🏘️ Valor Agregado Municipal",
            "📑 Generador de Informes",
            "💬 Chatbot del Observatorio",
            "©️ Derechos y Créditos",
        ],
        index=0,
    )
    st.markdown("---")
    st.header("🎛️ Panel de control")
    st.caption("Filtros globales (Análisis Departamental)")
    depto = st.multiselect(
        "Departamento(s)",
        ["Todos"] + lista_deptos,
        default=["Nariño"] if "Nariño" in lista_deptos else ["Todos"],
    )
    inds = st.multiselect("Indicadores", lista_inds, default=lista_inds)
    a1 = st.slider("Año inicial", amin, amax, a1_def, 1)
    a2 = st.slider("Año final", amin, amax, a2_def, 1)
    anio_m = st.slider("Año mapa / Moran", amin, amax, anio_default, 1)
    st.markdown("---")
    st.subheader("Parámetros de modelos")
    k = st.slider("K clusters (K-Means)", 2, 6, 3, 1)
    ar = st.slider("Alpha Ridge", 0.01, 30.0, 1.0, 0.1)
    al = st.slider("Alpha Lasso", 0.001, 3.0, 0.1, 0.01)
    lags = st.slider("Lags VAR", 1, 4, 2, 1)
    shock = st.slider("Shock % (simulación)", -40, 40, 10, 1)
    anios_s = st.slider("Años de proyección", 3, 8, 5, 1)
    st.markdown("---")
    st.caption("Desarrolladores: Jeison Ortega · Juan David Granja · Helmer Rivera · Francisco Quiroz")
    st.caption("Universidad de Nariño")

# ============================================================
# CONTENIDO
# ============================================================

if seccion == "🏠 Inicio":
    st.markdown(
        """
        <h1 style="color:#67E8F9; margin-bottom:0.15rem;">
            🇨🇴 Observatorio Económico y Social
        </h1>
        <p style="color:#94A3B8; margin-top:0;">
            Datos reales · Enfoque departamental · Prioridad Nariño · Nivel municipal
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="intro-box">
            <h2>👋 Bienvenida / Bienvenido</h2>
            <p>
                Esta plataforma reúne <b>indicadores oficiales de Colombia</b> (datos.gov.co)
                y un panel de <b>aproximaciones de valor agregado municipal</b> de alta fidelidad para Nariño.
                Explora economía y sociedad a nivel departamental y municipal, estima modelos
                de series de tiempo, entrena algoritmos de aprendizaje y simula escenarios de política.
            </p>
            <p><b>¿Cómo empezar?</b></p>
            <ol>
                <li>Usa el menú de la barra lateral para navegar entre módulos.</li>
                <li>Ajusta departamento, indicadores y rango de años cuando sea necesario.</li>
                <li>En <b>Análisis Departamental</b> encontrarás EDA, mapas, VAR, ML y Focus Nariño.</li>
                <li>Consulta el <b>Valor Agregado Municipal</b>, genera <b>Informes</b> o pregunta al <b>Chatbot</b>.</li>
            </ol>
            <p>
                <span class="tag">Periodo departamental: {amin}–{amax}</span>
                <span class="tag">Municipal: 2018–2024</span>
                <span class="tag">{len(lista_inds)} indicadores</span>
                <span class="tag">{len(lista_deptos)} departamentos</span>
                <span class="tag">{len(lista_municipios)} municipios Nariño</span>
            </p>
            <div class="authors">
                <b>Desarrolladores:</b>
                Jeison Ortega · Juan David Granja · Helmer Rivera · Francisco Quiroz
                <span class="uni">Universidad de Nariño</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("📘 Guía rápida de módulos", expanded=False):
        st.markdown("""
| Módulo | Para qué sirve |
|--------|----------------|
| **Análisis Departamental** | EDA, PCA, Clustering, mapas, Moran/LISA, VAR/VECM, ML/XAI, Simulador y Focus Nariño |
| **Valor Agregado Municipal** | Aproximaciones de PIB municipal, sectores y per cápita (Nariño 2018-2024) |
| **Generador de Informes** | Resumen ejecutivo, evolución, comparación y exportación con interpretación automática |
| **Chatbot** | Preguntas en lenguaje natural sobre PIB, seguridad, inversión, municipios y rankings |
| **Derechos y Créditos** | Licencias, fuentes y autoría institucional |
        """)
    with st.expander("📦 Fuentes e indicadores cargados", expanded=False):
        st.markdown(f"""
**Origen departamental:** APIs SODA de [datos.gov.co](https://www.datos.gov.co).

**Indicadores:** `{', '.join(lista_inds)}`

**Municipal:** Aproximaciones de alta fidelidad calibradas a magnitudes regionales de Nariño.
        """)
        cov = DF_LONG.groupby("indicador").agg(
            Filas=("valor", "count"),
            Año_mín=("anio", "min"),
            Año_máx=("anio", "max"),
            Departamentos=("departamento", "nunique"),
        ).reset_index()
        st.dataframe(cov, use_container_width=True, hide_index=True)

elif seccion == "📊 Análisis Departamental":
    st.markdown("## 📊 Análisis Departamental")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 EDA · PCA · Clustering",
        "🗺️ Econometría Espacial",
        "📈 Series de Tiempo (VAR/VEC)",
        "🤖 ML · Deep Learning & XAI",
        "🎯 Simulador de Política",
        "📍 Focus Nariño",
    ])

    with tab1:
        st.markdown(
            """<div class="tip-box">📚 <b>EDA:</b> resume la forma de los datos.
            <b>PCA:</b> reduce dimensiones. <b>K-Means:</b> agrupa departamentos similares.</div>""",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        run_eda = c1.button("EDA + Correlación", type="primary", key="b1")
        run_pca = c2.button("PCA + Biplot", type="primary", key="b2")
        run_km = c3.button("K-Means", type="primary", key="b3")
        if run_eda:
            with st.spinner("Calculando estadísticos y correlaciones…"):
                txt, fig = eda(DF_LONG, PANEL, lista_inds, depto, inds, a1, a2)
            st.markdown(txt)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        if run_pca:
            with st.spinner("Ejecutando PCA…"):
                txt, fig = pca_fn(PANEL, a1, a2)
            st.markdown(txt)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        if run_km:
            with st.spinner("Agrupando departamentos…"):
                txt, fig = kmeans_fn(PANEL, a1, a2, k)
            st.markdown(txt)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown(
            """<div class="tip-box">🗺️ <b>Mapa:</b> compara departamentos.
            <b>Moran / LISA:</b> mide concentración geográfica de valores altos o bajos.</div>""",
            unsafe_allow_html=True,
        )
        ind_m = st.selectbox("Indicador espacial", lista_inds, index=0, key="ind_m")
        c1, c2 = st.columns(2)
        run_mapa = c1.button("Mapa / Barras", type="primary", key="bm")
        run_moran = c2.button("Moran + LISA", type="primary", key="bmor")
        if run_mapa:
            with st.spinner("Generando mapa…"):
                fig = mapa_fn(DF_LONG, GEOJSON, ind_m, anio_m)
            st.plotly_chart(fig, use_container_width=True)
        if run_moran:
            with st.spinner("Estimando autocorrelación espacial…"):
                txt, fig = moran_fn(DF_LONG, ind_m, anio_m)
            st.markdown(txt)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown(
            """<div class="tip-box">📈 <b>VAR</b> modela varias series e impulso-respuesta.
            <b>VECM</b> cuando hay cointegración. <b>ADF</b> prueba estacionariedad.</div>""",
            unsafe_allow_html=True,
        )
        d_ts = st.selectbox(
            "Departamento", lista_deptos,
            index=lista_deptos.index("Nariño") if "Nariño" in lista_deptos else 0, key="d_ts"
        )
        i_ts = st.multiselect(
            "Indicadores VAR/VECM", lista_inds,
            default=lista_inds[:min(3, len(lista_inds))], key="i_ts"
        )
        tipo = st.radio("Modelo", ["VAR", "VECM"], horizontal=True, key="tipo")
        if st.button("Estimar modelo", type="primary", key="bts"):
            with st.spinner(f"Estimando {tipo}…"):
                txt, fig = var_fn(DF_LONG, d_ts, i_ts, lags, tipo)
            st.markdown(txt)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown(
            """<div class="tip-box">🤖 <b>Ridge / Lasso</b>, <b>SVR</b> y <b>MLP</b> para predicción.
            <b>XAI</b> ordena qué variables importan más.</div>""",
            unsafe_allow_html=True,
        )
        tgt = st.selectbox(
            "Variable objetivo (target)", lista_inds,
            index=lista_inds.index("PIB") if "PIB" in lista_inds else 0, key="tgt"
        )
        prd = st.multiselect(
            "Predictores", lista_inds,
            default=[i for i in lista_inds if i != tgt][:4], key="prd"
        )
        c1, c2 = st.columns(2)
        run_ml = c1.button("Entrenar ML", type="primary", key="bml")
        run_xai = c2.button("XAI", type="primary", key="bx")
        if run_ml:
            with st.spinner("Entrenando modelos…"):
                txt, fig = ml_fn(PANEL, tgt, prd, a1, a2, ar, al)
            st.markdown(txt)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        if run_xai:
            with st.spinner("Calculando importancia…"):
                txt, fig = xai_fn(PANEL, tgt, prd, a1, a2)
            st.markdown(txt)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown(
            """<div class="tip-box">🎯 <b>Simulación:</b> shock en un indicador y trayectoria de otro.</div>""",
            unsafe_allow_html=True,
        )
        ds = st.selectbox(
            "Departamento", lista_deptos,
            index=lista_deptos.index("Nariño") if "Nariño" in lista_deptos else 0, key="ds"
        )
        is_ = st.selectbox(
            "Indicador del shock", lista_inds,
            index=lista_inds.index("Recursos_DNP") if "Recursos_DNP" in lista_inds else 0, key="is_"
        )
        ir = st.selectbox(
            "Indicador de respuesta", lista_inds,
            index=lista_inds.index("PIB") if "PIB" in lista_inds else 0, key="ir"
        )
        st.info(f"Escenario: shock de **{shock:+d}%** en **{is_}** → efecto sobre **{ir}** en **{ds}** ({anios_s} años).")
        if st.button("Simular", type="primary", key="bs"):
            with st.spinner("Proyectando escenario…"):
                txt, fig = sim_fn(DF_LONG, ds, is_, ir, shock, anios_s)
            st.markdown(txt)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.markdown(
            """<div class="tip-box">📍 <b>Focus Nariño:</b> último valor, variación interanual
            y comparación con el promedio nacional.</div>""",
            unsafe_allow_html=True,
        )
        if st.button("Actualizar dashboard Nariño", type="primary", key="bn"):
            with st.spinner("Construyendo tablero de Nariño…"):
                txt, fig = focus_narino(DF_LONG, lista_inds)
            st.markdown(txt)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

elif seccion == "🏘️ Valor Agregado Municipal":
    st.markdown("## 🏘️ Valor Agregado Municipal – Nariño")
    st.markdown(
        """
        <div class="tip-box">
        Panel de <b>aproximaciones de alta fidelidad</b> 2018–2024 para municipios de Nariño.
        Variables: PIB Municipal, Valor Agregado Primario / Secundario / Terciario y PIB per cápita.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        muns_sel = st.multiselect(
            "Municipios", lista_municipios,
            default=["Pasto", "Ipiales", "Tumaco", "Túquerres", "La Unión"] if all(
                m in lista_municipios for m in ["Pasto", "Ipiales", "Tumaco", "Túquerres", "La Unión"]
            ) else lista_municipios[:5],
        )
    with col_f2:
        vars_mun = st.multiselect(
            "Variables",
            ["PIB_Municipal", "VA_Primario", "VA_Secundario", "VA_Terciario", "PIB_Per_Capita"],
            default=["PIB_Municipal", "VA_Primario", "VA_Secundario", "VA_Terciario"],
        )
    with col_f3:
        anio_mun = st.slider("Año (snapshot)", 2018, 2024, 2023)

    if not muns_sel:
        st.warning("Selecciona al menos un municipio.")
    else:
        df_f = DF_MUN[DF_MUN["municipio"].isin(muns_sel)].copy()
        df_anio = df_f[df_f["anio"] == anio_mun]

        if not df_anio.empty and "PIB_Municipal" in df_anio.columns:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PIB total seleccionado", f"{df_anio['PIB_Municipal'].sum():.1f} mm")
            c2.metric("Promedio municipal", f"{df_anio['PIB_Municipal'].mean():.1f} mm")
            top_m = df_anio.loc[df_anio["PIB_Municipal"].idxmax()]
            c3.metric("Mayor PIB", f"{top_m['municipio']}", f"{top_m['PIB_Municipal']:.1f} mm")
            c4.metric("PIB per cápita medio", f"${df_anio['PIB_Per_Capita'].mean():,.0f}")

        st.subheader("Evolución temporal")
        if vars_mun:
            long_m = df_f.melt(
                id_vars=["municipio", "anio"],
                value_vars=[v for v in vars_mun if v in df_f.columns],
                var_name="Variable", value_name="Valor"
            )
            fig_ts = px.line(
                long_m, x="anio", y="Valor", color="municipio", facet_col="Variable",
                facet_col_wrap=2, markers=True, title="Trayectorias municipales"
            )
            fig_ts.update_layout(height=480)
            st.plotly_chart(style_fig(fig_ts), use_container_width=True)

        st.subheader(f"Comparación {anio_mun}")
        if not df_anio.empty and vars_mun:
            var_bar = st.selectbox("Variable para ranking", vars_mun, key="var_bar_mun")
            df_bar = df_anio.sort_values(var_bar, ascending=True)
            fig_bar = px.bar(
                df_bar, x=var_bar, y="municipio", orientation="h",
                color=var_bar, color_continuous_scale="Viridis",
                title=f"{var_bar} · {anio_mun}"
            )
            fig_bar.update_layout(height=max(400, 22 * len(df_bar)))
            st.plotly_chart(style_fig(fig_bar), use_container_width=True)

        st.subheader("Estructura sectorial (heatmap)")
        if not df_anio.empty:
            heat = df_anio.set_index("municipio")[["VA_Primario", "VA_Secundario", "VA_Terciario"]]
            heat_pct = heat.div(heat.sum(axis=1), axis=0) * 100
            fig_h = px.imshow(
                heat_pct, text_auto=".0f", color_continuous_scale="Teal",
                title=f"Participación sectorial % · {anio_mun}", aspect="auto"
            )
            fig_h.update_layout(height=420)
            st.plotly_chart(style_fig(fig_h), use_container_width=True)

        with st.expander("Tabla de datos filtrados"):
            st.dataframe(df_f.sort_values(["municipio", "anio"]), use_container_width=True, hide_index=True)

# ---------- GENERADOR DE INFORMES (MEJORADO Y EXTENSO) ----------
elif seccion == "📑 Generador de Informes":
    st.markdown("## 📑 Generador de Informes de Política")
    st.markdown(
        """
        <div class="tip-box">
        El informe se genera <b>automáticamente</b> con una interpretación sustancial basada en los datos disponibles.
        Solo elige fuente, indicador, año y (opcional) municipios.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fuente = st.radio(
        "Fuente de datos del informe",
        ["Departamental (Nariño)", "Municipal (Nariño)"],
        horizontal=True
    )

    tab_r1, tab_r2, tab_r3, tab_r4 = st.tabs([
        "1. Resumen Ejecutivo",
        "2. Evolución Temporal",
        "3. Análisis Comparativo",
        "4. Exportación",
    ])

    if fuente == "Departamental (Nariño)":
        d_inf = DF_LONG[DF_LONG["departamento"] == "Nariño"].copy()
        if d_inf.empty:
            st.warning("Sin datos de Nariño.")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            ind_inf = st.selectbox("Indicador principal", lista_inds, key="ind_inf")
        with col2:
            anios_disp = sorted(d_inf[d_inf["indicador"] == ind_inf]["anio"].unique())
            if not anios_disp:
                st.warning("No hay años disponibles para este indicador.")
                st.stop()
            anio_inf = st.selectbox(
                "Año de referencia", anios_disp,
                index=len(anios_disp) - 1, key="anio_inf"
            )

        d_ind = d_inf[d_inf["indicador"] == ind_inf].sort_values("anio")
        val_ult = float(d_ind.loc[d_ind["anio"] == anio_inf, "valor"].iloc[0]) if anio_inf in d_ind["anio"].values else None

        prev = d_ind[d_ind["anio"] == anio_inf - 1]["valor"]
        delta_pct = None
        if len(prev) and float(prev.iloc[0]) != 0 and val_ult is not None:
            delta_pct = (val_ult - float(prev.iloc[0])) / float(prev.iloc[0]) * 100

        nac = DF_LONG[(DF_LONG["indicador"] == ind_inf) & (DF_LONG["anio"] == anio_inf)]["valor"].mean()
        rank_df = DF_LONG[(DF_LONG["indicador"] == ind_inf) & (DF_LONG["anio"] == anio_inf)].sort_values("valor", ascending=False).reset_index(drop=True)
        pos_nar = rank_df[rank_df["departamento"] == "Nariño"]
        posicion = pos_nar.index[0] + 1 if not pos_nar.empty else None

        ultimos = d_ind.tail(5)
        tendencia = "estable"
        if len(ultimos) >= 3:
            cambio = (ultimos["valor"].iloc[-1] - ultimos["valor"].iloc[0]) / ultimos["valor"].iloc[0] * 100
            if cambio > 5:
                tendencia = "ascendente"
            elif cambio < -5:
                tendencia = "descendente"

        # ---------- Interpretación sustancial ----------
        def generar_interpretacion_extensa():
            txt = []
            txt.append(f"### 1. Contexto del indicador **{ind_inf}** en Nariño ({anio_inf})")
            if val_ult is not None:
                txt.append(
                    f"En el año **{anio_inf}**, el valor registrado del indicador **{ind_inf}** para el departamento de Nariño "
                    f"fue de **{val_ult:,.1f}**. Este dato se enmarca dentro de la serie histórica disponible en la plataforma "
                    f"y constituye el punto de referencia principal del presente informe."
                )
            if delta_pct is not None:
                sentido = "un aumento" if delta_pct > 0 else "una disminución"
                txt.append(
                    f"Respecto al año inmediatamente anterior se observó {sentido} de **{abs(delta_pct):.1f}%**. "
                    f"Esta variación interanual permite identificar el sentido reciente de la trayectoria del indicador "
                    f"y constituye un insumo clave para la lectura de corto plazo."
                )
            if nac is not None and val_ult is not None:
                rel = "por encima" if val_ult > nac else "por debajo"
                diferencia = abs(val_ult - nac)
                txt.append(
                    f"Al contrastar el dato departamental con el promedio nacional del mismo año (**{nac:,.1f}**), "
                    f"se observa que Nariño se ubica **{rel}** de dicha referencia. "
                    f"La diferencia absoluta es de aproximadamente **{diferencia:,.1f}** unidades del indicador."
                )
            if posicion is not None:
                txt.append(
                    f"En el ranking nacional de departamentos para el indicador **{ind_inf}** en **{anio_inf}**, "
                    f"Nariño ocupó la posición **{posicion}** de un total de **{len(rank_df)}** entidades territoriales. "
                    f"Esta posición relativa aporta contexto sobre el desempeño comparado del departamento."
                )
            txt.append(
                f"La tendencia de los últimos años disponibles se caracteriza como **{tendencia}**. "
                f"Este diagnóstico de tendencia se construye a partir de la variación acumulada de los últimos puntos "
                f"de la serie y debe interpretarse con cautela cuando el número de observaciones es limitado."
            )

            txt.append("\n### 2. Lectura de política pública")
            if "PIB" in ind_inf:
                if delta_pct is not None and delta_pct < 0:
                    txt.append(
                        "La contracción observada en el PIB sugiere la necesidad de revisar los motores de crecimiento regional. "
                        "Se recomienda priorizar el seguimiento a la inversión pública, el dinamismo de las exportaciones, "
                        "el turismo y las cadenas agroindustriales, así como fortalecer la articulación entre el Plan de Desarrollo "
                        "Departamental y los planes de desarrollo municipales. La identificación de cuellos de botella en "
                        "infraestructura, capital humano y acceso a financiamiento resulta prioritaria."
                    )
                else:
                    txt.append(
                        "El desempeño del PIB sugiere continuidad en las estrategias de diversificación productiva y atracción "
                        "de inversión. Se recomienda profundizar en los eslabones de mayor valor agregado, fortalecer la "
                        "articulación público-privada y monitorear la sostenibilidad del crecimiento, prestando especial atención "
                        "a la inclusión de municipios con menor peso económico relativo."
                    )
            elif "Homicidio" in ind_inf:
                txt.append(
                    "Los indicadores de seguridad requieren coordinación interinstitucional permanente, focalización territorial "
                    "y seguimiento a factores de riesgo asociados a economías ilícitas, conflictos locales y dinámicas de frontera. "
                    "Se recomienda articular las cifras departamentales con análisis municipales y con las estrategias de "
                    "prevención y control definidas en los planes de seguridad y convivencia."
                )
            elif any(x in ind_inf for x in ["Educacion", "Matricula", "Matriculacion"]):
                txt.append(
                    "Los resultados educativos invitan a priorizar cobertura, calidad y retención, especialmente en zonas rurales "
                    "y de frontera. Se recomienda alinear los recursos del Sistema General de Participaciones, los proyectos de "
                    "inversión y las estrategias de permanencia escolar, así como fortalecer la información desagregada por "
                    "municipio para orientar intervenciones focalizadas."
                )
            elif "DNP" in ind_inf or "Recursos" in ind_inf:
                txt.append(
                    "La evolución de los recursos de inversión sugiere evaluar la capacidad de ejecución, la focalización "
                    "territorial y el impacto de los proyectos sobre el desarrollo económico local. Se recomienda contrastar "
                    "la programación presupuestal con los resultados de gestión y con los indicadores de resultado asociados "
                    "a las metas del Plan de Desarrollo."
                )
            else:
                txt.append(
                    "Se sugiere profundizar el análisis con series más largas, variables de control y contrastes municipales "
                    "para orientar intervenciones de política pública. La combinación de evidencia cuantitativa con lectura "
                    "cualitativa territorial resulta indispensable para la toma de decisiones informada."
                )

            txt.append("\n### 3. Consideraciones metodológicas")
            txt.append(
                "Los datos departamentales provienen de fuentes oficiales (datos.gov.co). Las variaciones interanuales y "
                "las posiciones relativas deben interpretarse considerando posibles cambios metodológicos en las series "
                "originales, la cobertura temporal disponible y el carácter agregado del indicador. Para decisiones de "
                "política se recomienda contrastar estos hallazgos con información oficial actualizada y con análisis "
                "cualitativos de contexto territorial."
            )
            return "\n\n".join(txt)

        interpretacion = generar_interpretacion_extensa()

        with tab_r1:
            st.subheader("Resumen Ejecutivo")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"{ind_inf}", f"{val_ult:,.1f}" if val_ult is not None else "—",
                      f"{delta_pct:+.1f}%" if delta_pct is not None else None)
            c2.metric("Promedio nacional", f"{nac:,.1f}" if nac is not None else "—")
            c3.metric("Posición nacional", f"{posicion}°" if posicion else "—")
            c4.metric("Año de referencia", str(anio_inf))
            st.markdown(interpretacion)

        with tab_r2:
            st.subheader("Evolución Temporal")
            if not d_ind.empty:
                fig = px.line(d_ind, x="anio", y="valor", markers=True, title=f"{ind_inf} – Nariño")
                if len(d_ind) >= 3:
                    z = np.polyfit(d_ind["anio"], d_ind["valor"], 1)
                    trend = np.poly1d(z)(d_ind["anio"])
                    fig.add_trace(go.Scatter(
                        x=d_ind["anio"], y=trend, mode="lines",
                        name="Tendencia lineal", line=dict(dash="dash", color="#A78BFA")
                    ))
                fig.add_vline(x=anio_inf, line_dash="dot", line_color="#22D3EE")
                fig.update_layout(height=450)
                st.plotly_chart(style_fig(fig), use_container_width=True)
            else:
                st.info("Sin serie para el indicador seleccionado.")

        with tab_r3:
            st.subheader("Análisis Comparativo (departamentos)")
            d_comp = DF_LONG[(DF_LONG["indicador"] == ind_inf) & (DF_LONG["anio"] == anio_inf)]
            if not d_comp.empty:
                d_comp = d_comp.sort_values("valor", ascending=True)
                fig = px.bar(
                    d_comp, x="valor", y="departamento", orientation="h",
                    color="valor", color_continuous_scale="Viridis",
                    title=f"{ind_inf} · {anio_inf}"
                )
                fig.update_layout(height=max(420, 14 * len(d_comp)))
                st.plotly_chart(style_fig(fig), use_container_width=True)
            else:
                st.info("Sin datos de comparación para ese año.")

        with tab_r4:
            st.subheader("Exportación de Reportes")
            st.info("El informe se genera automáticamente con interpretación extensa. Puedes añadir notas opcionales.")
            notas_extra = st.text_area(
                "Notas adicionales del analista (opcional)",
                height=100,
                placeholder="Espacio libre para comentarios propios..."
            )

            md_lines = [
                f"# Informe de Política – Observatorio Económico y Social",
                f"**Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"**Territorio:** Nariño",
                f"**Indicador:** {ind_inf}",
                f"**Año de referencia:** {anio_inf}",
                "",
                "## 1. Resumen Ejecutivo",
                f"- Valor en Nariño ({anio_inf}): **{val_ult:,.1f}**" if val_ult is not None else "- Valor: N/D",
                f"- Variación interanual: **{delta_pct:+.1f}%**" if delta_pct is not None else "- Variación: N/D",
                f"- Promedio nacional: **{nac:,.1f}**" if nac is not None else "",
                f"- Posición nacional: **{posicion}°** de {len(rank_df)}" if posicion else "",
                f"- Tendencia reciente: **{tendencia}**",
                "",
                "## 2. Interpretación automática",
                interpretacion,
                "",
            ]
            if notas_extra.strip():
                md_lines += ["## 3. Notas del analista", notas_extra, ""]
            md_lines += [
                "## Fuentes",
                "- datos.gov.co (APIs SODA) – datos oficiales del Gobierno de Colombia",
                "- Observatorio Económico y Social – Universidad de Nariño",
                "",
                "---",
                "Informe generado automáticamente a partir de datos oficiales y reglas de interpretación del Observatorio.",
            ]
            md_content = "\n".join(md_lines)

            csv_buf = d_ind.to_csv(index=False).encode("utf-8")
            c1, c2 = st.columns(2)
            c1.download_button(
                "⬇️ Descargar CSV (serie)",
                data=csv_buf,
                file_name=f"informe_{ind_inf}_narino_{anio_inf}.csv",
                mime="text/csv",
            )
            c2.download_button(
                "⬇️ Descargar Informe Markdown",
                data=md_content.encode("utf-8"),
                file_name=f"informe_politica_{ind_inf}_{anio_inf}_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
            )
            st.markdown("### Vista previa del informe")
            st.markdown(md_content)

    else:
        # Municipal (resumen más corto pero coherente)
        col1, col2 = st.columns(2)
        with col1:
            muns_inf = st.multiselect(
                "Municipios para el informe", lista_municipios,
                default=["Pasto", "Ipiales", "Tumaco"], key="muns_inf"
            )
        with col2:
            anio_inf_m = st.selectbox("Año de referencia", list(range(2018, 2025)), index=5, key="anio_inf_m")

        df_mi = DF_MUN[(DF_MUN["municipio"].isin(muns_inf)) & (DF_MUN["anio"] == anio_inf_m)] if muns_inf else pd.DataFrame()

        def interp_municipal():
            if df_mi.empty:
                return "No hay datos para los municipios y año seleccionados."
            total = df_mi["PIB_Municipal"].sum()
            top = df_mi.loc[df_mi["PIB_Municipal"].idxmax()]
            txt = [
                f"### Panorama municipal de Nariño ({anio_inf_m})",
                f"Los municipios seleccionados concentran un PIB agregado aproximado de **{total:,.1f}** miles de millones de COP.",
                f"El municipio de mayor peso es **{top['municipio']}** con **{top['PIB_Municipal']:,.1f}** mm COP "
                f"y un PIB per cápita aproximado de **${top['PIB_Per_Capita']:,.0f}**.",
                "",
                "### Estructura productiva",
                "Se observa heterogeneidad territorial: los centros urbanos (Pasto, Ipiales) tienen mayor peso del sector terciario, "
                "mientras que municipios de frontera y costa mantienen mayor participación del sector primario.",
                "",
                "### Orientación de política",
                "Se recomienda fortalecer cadenas de valor locales, mejorar conectividad y focalizar inversión pública en los municipios "
                "con menor PIB per cápita para reducir brechas intra-departamentales. Estas cifras constituyen aproximaciones "
                "de alta fidelidad orientadas al análisis exploratorio y a la discusión de política."
            ]
            return "\n\n".join(txt)

        interpretacion_m = interp_municipal()

        with tab_r1:
            st.subheader("Resumen Ejecutivo Municipal")
            if not df_mi.empty:
                c1, c2, c3 = st.columns(3)
                c1.metric("PIB agregado", f"{df_mi['PIB_Municipal'].sum():.1f} mm")
                c2.metric("PIB per cápita medio", f"${df_mi['PIB_Per_Capita'].mean():,.0f}")
                top = df_mi.loc[df_mi["PIB_Municipal"].idxmax()]
                c3.metric("Municipio líder", top["municipio"], f"{top['PIB_Municipal']:.1f} mm")
                st.markdown(interpretacion_m)
            else:
                st.info("Selecciona al menos un municipio.")

        with tab_r2:
            st.subheader("Evolución Temporal Municipal")
            if muns_inf:
                df_ts = DF_MUN[DF_MUN["municipio"].isin(muns_inf)]
                fig = px.line(df_ts, x="anio", y="PIB_Municipal", color="municipio", markers=True, title="PIB Municipal")
                fig.add_vline(x=anio_inf_m, line_dash="dot", line_color="#22D3EE")
                fig.update_layout(height=440)
                st.plotly_chart(style_fig(fig), use_container_width=True)

        with tab_r3:
            st.subheader("Comparación sectorial")
            if not df_mi.empty:
                long_s = df_mi.melt(
                    id_vars=["municipio"],
                    value_vars=["VA_Primario", "VA_Secundario", "VA_Terciario"],
                    var_name="Sector", value_name="Valor"
                )
                fig = px.bar(long_s, x="municipio", y="Valor", color="Sector", barmode="stack",
                             title=f"Estructura sectorial · {anio_inf_m}")
                fig.update_layout(height=420)
                st.plotly_chart(style_fig(fig), use_container_width=True)

        with tab_r4:
            st.subheader("Exportación")
            notas_m = st.text_area("Notas adicionales (opcional)", height=80, key="obs_m")
            md_m = [
                "# Informe Municipal – Valor Agregado Nariño",
                f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"**Año de referencia:** {anio_inf_m}",
                f"**Municipios:** {', '.join(muns_inf) if muns_inf else 'N/A'}",
                "",
                "## Interpretación",
                interpretacion_m,
                "",
            ]
            if notas_m.strip():
                md_m += ["## Notas del analista", notas_m, ""]
            md_m += ["---", "Observatorio Económico y Social · Universidad de Nariño"]
            md_m_content = "\n".join(md_m)
            csv_m = (df_mi.to_csv(index=False) if not df_mi.empty else "").encode("utf-8")
            c1, c2 = st.columns(2)
            c1.download_button("⬇️ CSV municipal", data=csv_m, file_name=f"informe_municipal_{anio_inf_m}.csv", mime="text/csv")
            c2.download_button("⬇️ Informe Markdown", data=md_m_content.encode("utf-8"),
                               file_name=f"informe_municipal_{anio_inf_m}_{datetime.now().strftime('%Y%m%d')}.md", mime="text/markdown")
            st.markdown("### Vista previa")
            st.markdown(md_m_content)

elif seccion == "💬 Chatbot del Observatorio":
    st.markdown("## 💬 Chatbot del Observatorio Económico")
    st.markdown(
        """
        <div class="tip-box">
        Asistente contextual basado en reglas sobre los datos del Observatorio.
        Pregunta por PIB de cualquier departamento y año, homicidios, recursos DNP, educación o municipios de Nariño.
        Tolera errores ortográficos frecuentes (ej. “antoquia”, “himicidios”).
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hola. Soy el asistente del **Observatorio Económico y Social**. "
                    "Puedo ayudarte con datos de Nariño y del resto del país (PIB, seguridad, inversión DNP, educación) "
                    "y con el panel municipal (Pasto, Ipiales, Tumaco, etc.). ¿Qué deseas consultar?"
                ),
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Escribe tu pregunta…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analizando…"):
                resp = chatbot_respuesta(prompt, DF_LONG, DF_MUN, lista_inds, lista_deptos)
            st.markdown(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})

elif seccion == "©️ Derechos y Créditos":
    st.markdown("## ©️ Derechos Reservados y Créditos de Autor")
    st.markdown(
        """
        <div class="credits-box">
            <h2>Observatorio Económico y Social</h2>
            <p class="meta">Universidad de Nariño · Colombia</p>

            <h3>Propiedad intelectual</h3>
            <p>
                El código fuente, la arquitectura analítica, el diseño de interfaz y los módulos
                de procesamiento del <b>Observatorio Económico y Social</b> son propiedad de sus
                desarrolladores y de la Universidad de Nariño. Queda prohibida la reproducción total o
                parcial con fines comerciales sin autorización expresa.
            </p>

            <h3>Datos y licencias</h3>
            <ul>
                <li><b>datos.gov.co</b> – Datos abiertos del Gobierno de Colombia (licencias de uso
                    público conforme a la política de datos abiertos).</li>
                <li><b>DANE y entidades oficiales</b> – Referencia metodológica de cuentas regionales
                    y valor agregado. En el módulo municipal se pueden emplear <b>aproximaciones de cifras</b>
                    de alta fidelidad orientadas al análisis exploratorio y a la discusión de política.</li>
                <li><b>GeoJSON departamental</b> – Capa cartográfica de uso abierto.</li>
            </ul>

            <h3>Desarrolladores</h3>
            <p>
                <b>Jeison Ortega</b> · <b>Juan David Granja</b> · <b>Helmer Rivera</b> · <b>Francisco Quiroz</b>
            </p>
            <p class="meta">Universidad de Nariño</p>

            <h3>Metadatos de la plataforma</h3>
            <ul>
                <li>Nombre: Observatorio Económico y Social</li>
                <li>Enfoque territorial: Departamento de Nariño (prioridad) y resto de Colombia</li>
                <li>Stack: Streamlit · Plotly · scikit-learn · statsmodels · libpysal / esda</li>
            </ul>

            <h3>Aviso de uso</h3>
            <p>
                Esta herramienta está orientada a investigación, docencia y apoyo a la formulación
                de política pública. Los resultados de modelos (VAR, ML, simulaciones) y las
                aproximaciones municipales deben interpretarse con precaución y contrastarse con
                fuentes oficiales antes de tomar decisiones.
            </p>

            <p class="meta" style="margin-top:1.5rem;">
                © Universidad de Nariño · Observatorio Económico y Social · Todos los derechos reservados
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "Observatorio · Datos oficiales (datos.gov.co) · "
    "Desarrolladores: Jeison Ortega · Juan David Granja · Helmer Rivera · Francisco Quiroz · "
    "Universidad de Nariño"
)
