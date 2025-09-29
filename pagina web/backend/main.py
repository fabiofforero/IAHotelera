# =========================================
# Backend FastAPI para:
#  - /predict  (consume tu modelo)
#  - /hotels   (devuelve hoteles por segmento desde CSV)
#  - /metadata (feature_names y class_names)
# Reqs: pip install fastapi uvicorn joblib scikit-learn pandas numpy pydantic
# Run:  python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
# =========================================
from pathlib import Path
from typing import Optional, Literal, Dict, Any
import json
import math

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from starlette.responses import JSONResponse

# -------------------------------
# Rutas de archivos (Windows)
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MODEL_PATH    = DATA_DIR / "extratrees_obj3_pipeline.joblib"
METADATA_PATH = DATA_DIR / "metadata.json"
HOTELS_CSV    = DATA_DIR / "hoteles_tunja_catalogo.csv"

# -------------------------------
# App + CORS (un solo middleware)
# -------------------------------
app = FastAPI(title="Recomendador Hoteles Tunja", version="0.1.0")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # opcional si usas vite preview
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,   # si no necesitas cookies, puedes poner False
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Validación de archivos críticos
# -------------------------------
missing = []
if not MODEL_PATH.exists():    missing.append(str(MODEL_PATH))
if not METADATA_PATH.exists(): missing.append(str(METADATA_PATH))
if missing:
    raise FileNotFoundError("Faltan archivos obligatorios:\n" + "\n".join(missing))

# -------------------------------
# Cargar modelo + metadata
# -------------------------------
model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
FEATURES: list[str] = metadata.get("feature_names", [])
CLASS_NAMES: list[str] = metadata.get("class_names", ["económico", "precio-calidad", "premium"])

# -------------------------------
# Catálogo de hoteles
# -------------------------------
def cargar_catalogo() -> pd.DataFrame:
    if HOTELS_CSV.exists():
        df = pd.read_csv(HOTELS_CSV)
        # Normalización básica de columnas
        cols_lower = {c.lower(): c for c in df.columns}
        for req in ["nombre", "segmento"]:
            if req not in [c.lower() for c in df.columns]:
                raise ValueError(f"El catálogo necesita la columna '{req}'")
        out = pd.DataFrame()
        out["nombre"]   = df[cols_lower.get("nombre", "nombre")]
        out["segmento"] = df[cols_lower.get("segmento", "segmento")].astype(str).str.strip().str.lower()
        out["direccion"]       = df[cols_lower.get("direccion", "direccion")] if "direccion" in cols_lower else ""
        out["telefono"]        = df[cols_lower.get("telefono", "telefono")] if "telefono" in cols_lower else ""
        out["sitio_web"]       = df[cols_lower.get("sitio_web", "sitio_web")] if "sitio_web" in cols_lower else ""
        out["precio_promedio"] = df[cols_lower.get("precio_promedio", "precio_promedio")] if "precio_promedio" in cols_lower else ""
        out["rating"]          = df[cols_lower.get("rating", "rating")] if "rating" in cols_lower else ""
        return out
    # Catálogo placeholder si no hay CSV (5 por segmento)
    data = []
    for i in range(1, 6): data.append({"nombre": f"Hotel Económico {i}",      "segmento": "económico"})
    for i in range(1, 6): data.append({"nombre": f"Hotel Precio-Calidad {i}", "segmento": "precio-calidad"})
    for i in range(1, 6): data.append({"nombre": f"Hotel Premium {i}",        "segmento": "premium"})
    return pd.DataFrame(data)

CATALOGO = cargar_catalogo()

# === Stats de precio por segmento (para re-ponderar por precio si lo activas) ===
SEG_PRICE_MEAN: Dict[str, float] = {}
SEG_PRICE_STD: Dict[str, float] = {}
if "precio_promedio" in CATALOGO.columns:
    _tmp = CATALOGO.copy()
    _tmp["precio_promedio"] = pd.to_numeric(_tmp["precio_promedio"], errors="coerce")
    for seg, g in _tmp.groupby("segmento"):
        vals = g["precio_promedio"].dropna()
        if len(vals):
            SEG_PRICE_MEAN[seg] = float(vals.mean())
            SEG_PRICE_STD[seg]  = float(max(vals.std(ddof=0), 1.0))  # evita sigma=0

# === Normalización de precio para poblar cal_norm ===
CATALOGO["_precio_num"] = pd.to_numeric(CATALOGO.get("precio_promedio"), errors="coerce")
if CATALOGO["_precio_num"].notna().sum() >= 5:
    PRICE_MIN = float(np.nanpercentile(CATALOGO["_precio_num"], 5))
    PRICE_MAX = float(np.nanpercentile(CATALOGO["_precio_num"], 95))
else:
    PRICE_MIN = 50000.0
    PRICE_MAX = 400000.0

def price_to_cal_norm(precio: float) -> float:
    """
    Mapea un precio (COP) a [0,1] con min/max robustos.
    Nuevo: CARO→1, BARATO→0.
    """
    lo, hi = PRICE_MIN, PRICE_MAX
    if hi <= lo:
        return 0.5
    x = (float(precio) - lo) / (hi - lo)
    return float(np.clip(x, 0.0, 1.0))  # caro->1, barato->0

# -------------------------------
# Tipos y esquema de entrada
# -------------------------------
TipoViajero = Literal["Familia","Grupo","Individual","Pareja"]
MotivoViaje = Literal["Turismo","Trabajo","Eventos culturales", "Vacaciones"]
Temporada   = Literal["Vacaciones","Semana Santa","Puentes festivos","Navidad"]
Canal       = Literal["Booking","Airbnb","Agencia física","Directo"]
Ocupacion   = Literal["Alta","Media","Baja"]
RepOnline   = Literal["Alta","Media","Baja"]
Limpieza    = Literal["Regular","Buena","Excelente"]
Procedencia = Literal["Bogotá","EEUU","España","Medellín","México","Tunja"]

class PredictIn(BaseModel):
    # Permite poblar por nombre interno (ASCII) o por alias (con acentos)
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    # Numéricos (alias ⇄ nombres que manda el frontend)
    edad: int = Field(35, ge=18, le=95, alias="Edad")
    genero: int = Field(1, ge=0, le=1, alias="Género", description="0=femenino, 1=masculino")
    duracion_estadia: int = Field(3, ge=1, le=60, alias="Duración_estadía")
    anticipacion_reserva_dias: int = Field(7, ge=0, le=365, alias="Anticipación_reserva_días")
    calificacion: float = Field(4.2, ge=0, le=5, alias="Calificación")

    # Binarios
    wifi: int = Field(1, ge=0, le=1, alias="WiFi")
    parqueadero: int = Field(1, ge=0, le=1, alias="Parqueadero")
    spa_gimnasio: int = Field(0, ge=0, le=1, alias="Spa_gimnasio")
    restaurante: int = Field(1, ge=0, le=1, alias="Restaurante")
    desayuno_incluido: int = Field(1, ge=0, le=1, alias="Desayuno_incluido")
    sostenibilidad: int = Field(0, ge=0, le=1, alias="Sostenibilidad")
    petfriendly: int = Field(0, ge=0, le=1, alias="PetFriendly")
    fidelizacion: int = Field(0, ge=0, le=1, alias="Fidelización")

    # Categóricas
    tipo_viajero: TipoViajero = Field("Pareja", alias="Tipo_viajero")
    motivo_viaje: MotivoViaje = Field("Turismo", alias="Motivo_viaje")
    temporada: Temporada = Field("Vacaciones", alias="Temporada")
    canal_reserva: Canal = Field("Directo", alias="Canal_reserva")
    ocupacion: Ocupacion = Field("Media", alias="Ocupación")
    reputacion_online: RepOnline = Field("Media", alias="Reputación_online")
    limpieza: Limpieza = Field("Buena", alias="Limpieza")
    procedencia: Procedencia = Field("Bogotá", alias="Procedencia")

    # Opcionales
    atencion_personal: Optional[float] = Field(None, ge=0, le=5, alias="Atención_personal")
    reputacion_score: Optional[float] = Field(None, ge=0, le=5)
    limpieza_score: Optional[float] = Field(None, ge=0, le=5)
    indice_valor: Optional[float] = Field(None, ge=0, le=5)

    # NUEVO: precio objetivo introducido por el usuario (en COP)
    precio: Optional[float] = Field(None, ge=0, alias="Precio")

# -------------------------------
# Construcción del vector de features
# -------------------------------
NUMERIC_DEFAULT_NAN = {
    "Edad","Duración_estadía","Anticipación_reserva_días","Calificación",
    "Atención_personal","reputacion_score","limpieza_score",
    "cal_norm","reputacion_norm","limpieza_norm","indice_valor"
}
BINARY_AS_INT = {
    "WiFi","Parqueadero","Spa_gimnasio","Restaurante",
    "Desayuno_incluido","Sostenibilidad","PetFriendly","Fidelización",
    # por compatibilidad, si tu pipeline tuviera Spa/Gimnasio separados:
    "Spa","Gimnasio"
}
def build_feature_row(req: PredictIn) -> tuple[pd.DataFrame, dict]:
    row: Dict[str, Any] = {}
    for f in FEATURES:
        row[f] = (np.nan if f in NUMERIC_DEFAULT_NAN else 0)

    # Numéricos
    row["Edad"] = req.edad
    row["Género"] = req.genero
    row["Duración_estadía"] = req.duracion_estadia
    row["Anticipación_reserva_días"] = req.anticipacion_reserva_dias
    row["Calificación"] = req.calificacion
    if req.atencion_personal is not None: row["Atención_personal"] = req.atencion_personal
    if req.reputacion_score is not None:  row["reputacion_score"]  = req.reputacion_score
    if req.limpieza_score is not None:    row["limpieza_score"]    = req.limpieza_score
    if req.indice_valor is not None:      row["indice_valor"]      = req.indice_valor

    # Binarios
    def set_bin(colname: str, value: int):
        if colname in row:
            row[colname] = int(value)
    set_bin("WiFi", req.wifi); set_bin("Parqueadero", req.parqueadero)
    set_bin("Spa_gimnasio", req.spa_gimnasio); set_bin("Restaurante", req.restaurante)
    set_bin("Desayuno_incluido", req.desayuno_incluido); set_bin("Sostenibilidad", req.sostenibilidad)
    set_bin("PetFriendly", req.petfriendly); set_bin("Fidelización", req.fidelizacion)
    set_bin("Spa", req.spa_gimnasio); set_bin("Gimnasio", req.spa_gimnasio)

    # One-hot
    def set_oh(prefix: str, val: str):
        key = f"{prefix}_{val}"
        if key in row: row[key] = 1
    set_oh("Tipo_viajero", req.tipo_viajero)
    set_oh("Motivo_viaje", req.motivo_viaje)
    set_oh("Temporada", req.temporada)
    set_oh("Canal_reserva", req.canal_reserva)
    set_oh("Ocupación", req.ocupacion)
    set_oh("Reputación_online", req.reputacion_online)
    set_oh("Limpieza", req.limpieza)

    # Procedencia
    proc_map = {"Bogotá":"Procedencia_Bogotá","EEUU":"Procedencia_EEUU","España":"Procedencia_España",
                "Medellín":"Procedencia_Medellín","México":"Procedencia_México","Tunja":"Procedencia_Tunja"}
    pkey = proc_map.get(req.procedencia)
    if pkey and pkey in row: row[pkey] = 1

    # === Derivadas / normalizaciones ===
    meta = {
        "precio_recibido": req.precio,
        "precio_usado": False,
        "cal_norm_fuente": None,
        "cal_norm_valor": None,
    }
    if (getattr(req, "precio", None) is not None) and ("cal_norm" in row):
        row["cal_norm"] = price_to_cal_norm(req.precio)
        meta.update(precio_usado=True, cal_norm_fuente="precio", cal_norm_valor=row["cal_norm"])
    else:
        if not np.isnan(row.get("Calificación", np.nan)) and ("cal_norm" in row):
            row["cal_norm"] = float(np.clip(row["Calificación"]/5.0, 0, 1))
            meta.update(precio_usado=False, cal_norm_fuente="calificacion", cal_norm_valor=row["cal_norm"])

    X = pd.DataFrame([[row[c] for c in FEATURES]], columns=FEATURES)
    return X, meta

# -------------------------------
# Re-ponderación por precio (opción "suave")
# -------------------------------
def reweight_by_price(proba_map: Dict[str, float], precio: Optional[float]) -> Dict[str, float]:
    """
    Aumenta la probabilidad de los segmentos cuyo precio medio
    esté más cerca del 'precio' deseado. Usa un peso Gaussiano.
    (No aplicada por defecto para evitar doble conteo con cal_norm)
    """
    if precio is None or not SEG_PRICE_MEAN:
        return proba_map

    weights: Dict[str, float] = {}
    for seg, pmean in SEG_PRICE_MEAN.items():
        sigma = SEG_PRICE_STD.get(seg, 1.0)
        w = math.exp(-((float(precio) - pmean) ** 2) / (2.0 * (sigma ** 2)))
        weights[seg] = w

    adj = {seg: proba_map.get(seg, 0.0) * weights.get(seg, 1.0) for seg in proba_map}
    s = sum(adj.values())
    if s > 0:
        adj = {k: v / s for k, v in adj.items()}
    return adj

# -------------------------------
# Endpoints
# -------------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "recomendador-hoteles-tunja", "version": "0.1.0"}

@app.get("/metadata")
def get_metadata():
    return {"feature_names": FEATURES, "class_names": CLASS_NAMES}

@app.get("/hotels")
def get_hotels(segment: Optional[str] = Query(None), limit: int = 5):
    df = CATALOGO.copy()
    if segment:
        seg = str(segment).strip().lower()
        df = df[df["segmento"] == seg]

    total = int(df.shape[0])
    if total == 0:
        return JSONResponse({"hoteles": [], "total": 0})

    # Muestra y saneo: Inf -> NaN, luego NaN -> None (JSON null)
    sample = df.sample(min(limit, total), random_state=None).reset_index(drop=True)
    sample = sample.replace([np.inf, -np.inf], np.nan)
    sample = sample.where(pd.notnull(sample), None)

    items = json.loads(sample.to_json(orient="records"))
    return JSONResponse({"hoteles": items, "total": total})

@app.post("/predict")
def predict(req: PredictIn):
    X, meta = build_feature_row(req)
    try:
        y_pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        proba_map = {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}

        segmento = (
            CLASS_NAMES[int(y_pred)]
            if isinstance(y_pred, (int, np.integer)) and int(y_pred) < len(CLASS_NAMES)
            else str(y_pred)
        )
        return {"segmento": segmento, "probabilidades": proba_map, "debug": meta}
    except Exception as e:
        raise HTTPException(500, detail=f"Error al predecir: {e}")
    
# en adelante segundo modelo 
# --- debajo de tus rutas BASE_DIR/DATA_DIR ---
DEMAND_MODEL_PATH = DATA_DIR / "modelo_knn_temporadas_personas.joblib"

try:
    demand_model = joblib.load(DEMAND_MODEL_PATH)
except Exception as e:
    demand_model = None
    print(f"[WARN] No pude cargar el modelo de demanda en {DEMAND_MODEL_PATH}: {e}")
from typing import List
from pydantic import BaseModel, Field

class DemandItem(BaseModel):
    temporada: str
    anio: int
    segmento: Literal["económico","precio-calidad","premium"]
    Tipo_viajero: Literal["Solo","Pareja","Familia","Grupo"]
    duracion_dias: int = Field(ge=1, le=60)
    mes_inicio: int = Field(ge=1, le=12)
    es_evento_ciudad: int = Field(0, ge=0, le=1)
    es_semana_santa: int = Field(0, ge=0, le=1)
    es_navidad: int = Field(0, ge=0, le=1)
    es_puente: int = Field(0, ge=0, le=1)

class DemandBatch(BaseModel):
    items: List[DemandItem]


def _norm_temporada(s: str) -> str:
    s = str(s).strip()
    return "Puente festivo" if s.lower().startswith("puente festivo") else s
@app.post("/predict_demand")
def predict_demand(batch: DemandBatch):
    if demand_model is None:
        raise HTTPException(500, "Modelo de demanda no cargado. Copia el .joblib en /data.")

    # → DataFrame crudo (la pipeline de PyCaret hace el preprocesado)
    df = pd.DataFrame([x.model_dump() for x in batch.items])
    if df.empty:
        return {"predictions": [], "totals": {}}

    # Normaliza rótulos de temporada
    df["temporada"] = df["temporada"].apply(_norm_temporada)

    # Predicción (personas)
    try:
        yhat = demand_model.predict(df)
    except Exception as e:
        raise HTTPException(500, f"Error al predecir demanda: {e}")

    df_out = df.copy()
    df_out["personas_pred"] = np.round(yhat).astype(int)

    # Agregados útiles
    totals = {
        "by_temporada": df_out.groupby("temporada")["personas_pred"].sum().to_dict(),
        "by_segmento": df_out.groupby("segmento")["personas_pred"].sum().to_dict(),
        "by_tipo": df_out.groupby("Tipo_viajero")["personas_pred"].sum().to_dict(),
    }

    # Sección detallada: temporada → segmento → lista por tipo
    det = {}
    for (temp, seg), g in df_out.groupby(["temporada","segmento"]):
        det.setdefault(temp, {})[seg] = (
            g[["Tipo_viajero","personas_pred"]]
            .sort_values("Tipo_viajero")
            .to_dict(orient="records")
        )

    return {
        "predictions": df_out.to_dict(orient="records"),
        "totals": totals,
        "detail": det
    }

