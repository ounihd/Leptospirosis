import io
import json
import logging
import numpy as np
import pandas as pd
import pickle
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import ee
import xgboost as xgb
from django.conf import settings
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from functools import lru_cache

import os
import json
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Earth Engine Initialization
# ---------------------------------------------------------------------------

_ee_initialized = False

def ensure_ee_initialized():
    if not ee.data._initialized:
        creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
        if creds_json:
            creds = service_account.Credentials.from_service_account_info(json.loads(creds_json))
            ee.Initialize(credentials=creds, project='ee-hd-leptospirosis')
        else:
            ee.Initialize(project='ee-hd-leptospirosis')

# ---------------------------------------------------------------------------
# ML Model Loading
# ---------------------------------------------------------------------------

_lepto_model = None

def load_lepto_model() -> Any:
    """Load the leptospirosis prediction model from XGBoost JSON file."""
    global _lepto_model
    if _lepto_model is None:
        model_path = Path(settings.BASE_DIR) / "lepto" / "ml" / "final_xgb_model.json"
        if not model_path.exists():
            logger.warning("Lepto model file not found: %s", model_path)
            return None
        try:
            _lepto_model = xgb.Booster()
            _lepto_model.load_model(str(model_path))
            logger.info("Loaded lepto model: %s", model_path)
        except Exception as e:
            logger.error("Failed to load lepto model: %s", e)
            _lepto_model = None
    return _lepto_model

def predict_lepto_risk(features_dict: Dict[str, Any]) -> Optional[float]:
    """
    Predict leptospirosis risk probability using the loaded XGBoost model.
    Features as per the specified list.
    Returns: Probability (0-1) or None if model unavailable/invalid.
    """
    model = load_lepto_model()
    if not model:
        return None

    try:
        # Replace None with np.nan and ensure float types
        features_dict = {k: float(v) if v is not None else np.nan for k, v in features_dict.items()}
        # Create DataFrame from features
        df = pd.DataFrame([features_dict])
        # Create DMatrix for XGBoost
        dmatrix = xgb.DMatrix(df)
        # Predict probability of positive class
        prediction = model.predict(dmatrix)[0]
        return float(prediction)
    except Exception as e:
        logger.error("Lepto prediction failed: %s", e)
        return None

# ---------------------------------------------------------------------------
# Population Density from Eurostat
# ---------------------------------------------------------------------------

BASE = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/DEMO_R_D3DENS"
WANTED_UNITS = {"P_KM2", "PER_KM2", "PERS_KM2"}  # persons per km²

def _to_number(x):
    if pd.isna(x): return np.nan
    x = str(x).strip()
    if x == ":": return np.nan
    x = x.split()[0].replace(",", "")
    return pd.to_numeric(x, errors="coerce")

def _df_for_year_csv(year: int) -> pd.DataFrame:
    r = requests.get(BASE, params={"time": str(year), "format": "SDMX-CSV"}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), dtype=str)
    cols = {c.lower(): c for c in df.columns}

    tidy = df.copy()
    if "freq" in cols: tidy = tidy[tidy[cols["freq"]].str.upper().eq("A")]
    if "unit" in cols: tidy = tidy[tidy[cols["unit"]].str.upper().isin(WANTED_UNITS)]
    if "time_period" in cols:
        tidy = tidy[tidy[cols["time_period"]].astype(str).str.strip().eq(str(year))]

    out = (
        tidy[[cols["geo"], cols["obs_value"]]]
        .rename(columns={cols["geo"]: "region", cols["obs_value"]: "value"})
        .assign(value=lambda d: d["value"].map(_to_number))
        .dropna(subset=["value"])
        .reset_index(drop=True)
    )
    return out

def get_population_density(region: str, year: int) -> float | None:
    df = _df_for_year_csv(int(year))
    s = df.loc[df["region"].str.upper().eq(region.upper()), "value"]
    if s.empty: return None
    v = s.iloc[0]
    return float(v) if pd.notna(v) else None

# ---------------------------------------------------------------------------
# GDP from Eurostat
# ---------------------------------------------------------------------------

BASE_GDP = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/NAMA_10R_3GDP"
WANTED_UNITS_GDP = {"MIO_EUR", "MIO_PPS"}   # million euro, million PPS
NA_ITEM_GDP = {"B1GQ"}  # if the dataset exposes NA_ITEM, keep GDP only

@lru_cache(maxsize=64)
def _df_gdp_year(year: int, unit: str = "MIO_EUR") -> pd.DataFrame:
    """
    Download + parse NAMA_10R_3GDP for one year + unit into tidy DataFrame
    with columns: region, value.
    """
    params = {"time": str(year), "format": "SDMX-CSV"}
    # Reduce payload by filtering unit server-side if the caller asked for a specific one
    if unit:
        params["unit"] = unit

    r = requests.get(BASE_GDP, params=params, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), dtype=str)

    cols = {c.lower(): c for c in df.columns}
    tidy = df

    # Annual frequency only if present
    if "freq" in cols:
        tidy = tidy[tidy[cols["freq"]].str.upper().eq("A")]

    # Keep GDP item if NA_ITEM is present (some exports don’t include it)
    if "na_item" in cols:
        tidy = tidy[tidy[cols["na_item"]].str.upper().isin(NA_ITEM_GDP)]

    # Keep requested unit if UNIT is present
    if "unit" in cols and unit:
        tidy = tidy[tidy[cols["unit"]].str.upper().eq(unit.upper())]
    elif "unit" in cols:
        tidy = tidy[tidy[cols["unit"]].str.upper().isin(WANTED_UNITS_GDP)]

    # Filter to the target year if TIME_PERIOD present
    if "time_period" in cols:
        tidy = tidy[tidy[cols["time_period"]].astype(str).str.strip().eq(str(year))]

    out = (
        tidy[[cols["geo"], cols["obs_value"]]]
        .rename(columns={cols["geo"]: "region", cols["obs_value"]: "value"})
        .assign(value=lambda d: d["value"].map(_to_number))
        .dropna(subset=["value"])
        .reset_index(drop=True)
    )
    return out

def get_gdp(region: str, year: int, unit: str = "MIO_EUR") -> float | None:
    """
    GDP at current market prices for a NUTS-3 region and year.

    Parameters
    ----------
    region : str   e.g. 'DE300' (Berlin), 'ITC11', 'ES511'
    year   : int   e.g. 2020
    unit   : str   'MIO_EUR' (default) or 'MIO_PPS'

    Returns
    -------
    float | None   GDP value in the requested unit, or None if not found.
    """
    df = _df_gdp_year(int(year), unit=unit)
    s = df.loc[df["region"].str.upper().eq(region.upper()), "value"]
    if s.empty:
        return None
    v = s.iloc[0]
    return float(v) if pd.notna(v) else None

# ---------------------------------------------------------------------------
# Employment from Eurostat
# ---------------------------------------------------------------------------

BASE_EMP = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/NAMA_10R_3EMPERS"

def _to_number(x):
    if pd.isna(x): return np.nan
    x = str(x).strip()
    if x == ":" or x == "": return np.nan
    x = x.split()[0].replace(",", "")
    return pd.to_numeric(x, errors="coerce")

@lru_cache(maxsize=64)
def _df_employment_year(year: int, unit: str = "THS", wstatus: str = "EMP", nace_r2: str = "TOTAL") -> pd.DataFrame:
    """
    Download + parse NAMA_10R_3EMPERS for one year into a tidy DataFrame
    with columns: region, value.
    """
    params = {
        "time": str(year),
        "format": "SDMX-CSV",
        "unit": unit,          # THS = thousand persons
        "wstatus": wstatus,    # EMP = employed persons
        "nace_r2": nace_r2,    # TOTAL = all NACE activities
    }

    r = requests.get(BASE_EMP, params=params, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text), dtype=str)
    cols = {c.lower(): c for c in df.columns}
    tidy = df

    # Annual only (if freq present)
    if "freq" in cols:
        tidy = tidy[tidy[cols["freq"]].str.upper().eq("A")]

    # Defensive filtering in case server ignored some params
    if "unit" in cols:
        tidy = tidy[tidy[cols["unit"]].str.upper().eq(unit.upper())]
    if "wstatus" in cols:
        tidy = tidy[tidy[cols["wstatus"]].str.upper().eq(wstatus.upper())]
    if "nace_r2" in cols:
        tidy = tidy[tidy[cols["nace_r2"]].str.upper().eq(nace_r2.upper())]
    if "time_period" in cols:
        tidy = tidy[tidy[cols["time_period"]].astype(str).str.strip().eq(str(year))]

    out = (
        tidy[[cols["geo"], cols["obs_value"]]]
        .rename(columns={cols["geo"]: "region", cols["obs_value"]: "value"})
        .assign(value=lambda d: d["value"].map(_to_number))
        .dropna(subset=["value"])
        .reset_index(drop=True)
    )
    return out

def get_employment(region: str, year: int, unit: str = "THS", wstatus: str = "EMP", nace_r2: str = "TOTAL") -> float | None:
    """
    Employment for a NUTS-3 region and year.
    Default unit = THS (thousand persons). Multiply by 1_000 for persons.
    """
    df = _df_employment_year(int(year), unit=unit, wstatus=wstatus, nace_r2=nace_r2)
    s = df.loc[df["region"].str.upper().eq(region.upper()), "value"]
    if s.empty:
        return None
    v = s.iloc[0]
    return float(v) if pd.notna(v) else None

# ---------------------------------------------------------------------------
# Bovine from Eurostat
# ---------------------------------------------------------------------------

BASE_BOVINE = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/AGR_R_ANIMAL"
CODELIST_GEO = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/codelist/ESTAT/GEO"

# Default unit: thousand heads (animals)
DEFAULT_UNIT_BOVINE = "THS_HD"

@lru_cache(maxsize=64)
def _df_bovine_year(year: int, animals: str, unit: str = DEFAULT_UNIT_BOVINE) -> pd.DataFrame:
    """
    Download + parse AGR_R_ANIMAL for one year into tidy df with columns: region (NUTS-2), value.
    """
    params = {
        "time": str(year),
        "format": "SDMX-CSV",
        "animals": animals,   # e.g. A2000=A: live bovine, A3100=swine, A4100=sheep, A4200=goats
        "unit": unit,         # THS_HD = thousand heads
    }
    r = requests.get(BASE_BOVINE, params=params, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text), dtype=str)
    cols = {c.lower(): c for c in df.columns}

    tidy = df
    if "freq" in cols:        # Annual only
        tidy = tidy[tidy[cols["freq"]].str.upper().eq("A")]
    if "time_period" in cols: # Target year
        tidy = tidy[tidy[cols["time_period"]].astype(str).str.strip().eq(str(year))]
    if "animals" in cols:     # Defensive in case server ignores filter
        tidy = tidy[tidy[cols["animals"]].str.upper().eq(animals.upper())]
    if "unit" in cols:
        tidy = tidy[tidy[cols["unit"]].str.upper().eq(unit.upper())]

    out = (
        tidy[[cols["geo"], cols["obs_value"]]]
        .rename(columns={cols["geo"]: "region", cols["obs_value"]: "value"})
        .assign(value=lambda d: d["value"].map(_to_number))
        .dropna(subset=["value"])
        .reset_index(drop=True)
    )
    # region here is NUTS-2
    return out

def _parent_nuts2(nuts3_code: str) -> str:
    """Assumes NUTS-3 codes are 5 chars; parent NUTS-2 is the first 4."""
    s = nuts3_code.strip().upper()
    if len(s) < 4:
        raise ValueError(f"Not a valid NUTS-3 code: {nuts3_code}")
    return s[:4]

@lru_cache(maxsize=1)
def _all_geo_codes() -> list[str]:
    """Fetch the GEO codelist and return all codes (strings)."""
    r = requests.get(CODELIST_GEO, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.text)

    codes = []
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]  # strip namespace
        if tag == "Code":
            cid = el.attrib.get("id") or el.attrib.get("value")
            if cid:
                codes.append(cid)
    return codes

@lru_cache(maxsize=1024)
def nuts3_children(nuts2_code: str) -> list[str]:
    """
    Return all NUTS-3 codes whose code starts with the given NUTS-2 code.
    (Heuristic using code prefix; works for NUTS 2021/2024 codes.)
    """
    pfx = nuts2_code.strip().upper()
    return sorted([c for c in _all_geo_codes() if len(c) == 5 and c.startswith(pfx)])

def get_bovine_nuts2(nuts2_code: str, year: int, animals: str, unit: str = DEFAULT_UNIT_BOVINE) -> float | None:
    """
    Bovine (thousand heads) for a NUTS-2 region and year.
    """
    df = _df_bovine_year(int(year), animals=animals, unit=unit)
    s = df.loc[df["region"].str.upper().eq(nuts2_code.upper()), "value"]
    if s.empty:
        return None
    v = s.iloc[0]
    return float(v) if pd.notna(v) else None

def get_bovine_nuts3(nuts3_code: str, year: int, animals: str, unit: str = DEFAULT_UNIT_BOVINE) -> float | None:
    """
    Bovine for a NUTS-3 code by reading its parent NUTS-2 value (same value mapped down).
    """
    parent = _parent_nuts2(nuts3_code)
    return get_bovine_nuts2(parent, year=year, animals=animals, unit=unit)

def get_bovine(region: str, year: int, animals: str = "A2000", unit: str = DEFAULT_UNIT_BOVINE) -> float | None:
    """
    Bovine (thousand heads) for a NUTS-3 region and year.
    Default animals = 'A2000' (live bovine animals).
    """
    return get_bovine_nuts3(region, year, animals, unit)

# ---------------------------------------------------------------------------
# Date and Window Helpers
# ---------------------------------------------------------------------------

def month_window(year: int, month: int) -> Tuple[str, str]:
    """Return start and end dates (YYYY-MM-DD) for the given month."""
    first = datetime(year, month, 1)
    if month == 12:
        last = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = datetime(year, month + 1, 1) - timedelta(days=1)
    return first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")

def get_prev_month_window(year_month: str) -> Tuple[str, str]:
    """Get previous month's window for 'YYYY-MM'."""
    year, month = map(int, year_month.split('-'))
    if month == 1:
        year -= 1
        month = 12
    else:
        month -= 1
    return month_window(year, month)

def get_three_months_prior_start_end(year_month: str) -> Tuple[str, str]:
    """Get three months prior window for 'YYYY-MM'."""
    year, month = map(int, year_month.split('-'))
    if month <= 3:
        year -= 1
        month += 9
    else:
        month -= 3
    return month_window(year, month)

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def safe_min(vals: list) -> Optional[float]:
    """Safe min of list, handling empty lists."""
    return min(vals) if vals else None

def safe_max(vals: list) -> Optional[float]:
    """Safe max of list, handling empty lists."""
    return max(vals) if vals else None

def range_diff(min_v: Optional[float], max_v: Optional[float]) -> Optional[float]:
    """Calculate range (max - min), handling None values."""
    if min_v is None or max_v is None:
        return None
    try:
        return float(max_v) - float(min_v)
    except (TypeError, ValueError):
        return None

# ---------------------------------------------------------------------------
# ERA5-Land Metrics
# ---------------------------------------------------------------------------

def compute_era5_stats(
    location: ee.Geometry,
    start_date: str,
    end_date: str,
    band: str,
    units: str
) -> Dict[str, Any]:
    """Generic ERA5 stats computation for a band."""
    start, end = ee.Date(start_date), ee.Date(end_date)

    monthly_col = (ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_BY_HOUR")
                   .filterBounds(location).filterDate(start, end).select(band))
    hourly_col = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
                  .filterBounds(location).filterDate(start, end).select(band))

    monthly_mean = monthly_col.mean().reduceRegion(
        reducer=ee.Reducer.mean(), geometry=location, scale=1000, maxPixels=1e8
    )

    def per_image_mean(img: ee.Image) -> ee.Image:
        mean_val = img.reduceRegion(ee.Reducer.mean(), location, 1000).get(band)
        return img.set("mean", mean_val)

    hourly_means = hourly_col.map(per_image_mean).aggregate_array("mean").getInfo() or []
    min_v, max_v = safe_min(hourly_means), safe_max(hourly_means)

    return {
        "monthly_mean": (monthly_mean.getInfo() or {}).get(band, None),
        "min_hourly_mean": min_v,
        "max_hourly_mean": max_v,
        "range_hourly_mean": range_diff(min_v, max_v),
        "hourly_count": len(hourly_means),
        "units": units,
    }

def prec(location: ee.Geometry, start_date: str, end_date: str) -> Dict[str, Any]:
    """Precipitation stats (ERA5-Land), meters of water (m)."""
    return compute_era5_stats(location, start_date, end_date, "total_precipitation", "m")

def temp(location: ee.Geometry, start_date: str, end_date: str) -> Dict[str, Any]:
    """Air temperature stats (ERA5-Land) in Kelvin (K)."""
    return compute_era5_stats(location, start_date, end_date, "temperature_2m", "K")

def soil_moisture(location: ee.Geometry, start_date: str, end_date: str) -> Dict[str, Any]:
    """Volumetric soil water (layer 1) — m³/m³."""
    return compute_era5_stats(location, start_date, end_date, "volumetric_soil_water_layer_1", "m3/m3")

def soil_temp(location: ee.Geometry, start_date: str, end_date: str) -> Dict[str, Any]:
    """Soil temperature (level 1) — Kelvin (K)."""
    return compute_era5_stats(location, start_date, end_date, "soil_temperature_level_1", "K")

# ---------------------------------------------------------------------------
# Landsat-7 Indices (NDWI/NDVI)
# ---------------------------------------------------------------------------

def compute_landsat_index(
    location: ee.Geometry,
    start_date: str,
    end_date: str,
    index_name: str,
    band_calc: callable
) -> Dict[str, Any]:
    """Generic Landsat-7 index computation."""
    start, end = ee.Date(start_date), ee.Date(end_date)
    col = (ee.ImageCollection("LANDSAT/LE07/C02/T1_TOA")
           .filterBounds(location).filterDate(start, end))

    def add_index(img: ee.Image) -> ee.Image:
        index_img = band_calc(img).rename(index_name)
        return img.addBands(index_img)

    with_index = col.map(add_index)
    mean_index = with_index.select(index_name).mean().reduceRegion(
        reducer=ee.Reducer.mean(), geometry=location, scale=30, maxPixels=1e8
    )

    return {
        "monthly_mean": (mean_index.getInfo() or {}).get(index_name, None),
        "units": "unitless",
    }

def ndwi(location: ee.Geometry, start_date: str, end_date: str) -> Dict[str, Any]:
    """NDWI from Landsat-7 TOA (B4 NIR, B5 SWIR1)."""
    def calc(img): return img.select("B4").subtract(img.select("B5")).divide(img.select("B4").add(img.select("B5")))
    return compute_landsat_index(location, start_date, end_date, "NDWI", calc)

def ndvi(location: ee.Geometry, start_date: str, end_date: str) -> Dict[str, Any]:
    """NDVI from Landsat-7 TOA (B4 NIR, B3 Red)."""
    def calc(img): return img.select("B4").subtract(img.select("B3")).divide(img.select("B4").add(img.select("B3")))
    return compute_landsat_index(location, start_date, end_date, "NDVI", calc)

# ---------------------------------------------------------------------------
# MODIS Land Cover
# ---------------------------------------------------------------------------

_LC_CLASS_NAMES = [
    'Water Bodies', 'Evergreen Needleleaf Vegetation', 'Evergreen Broadleaf Vegetation',
    'Deciduous Needleleaf Vegetation', 'Deciduous Broadleaf Vegetation', 'Annual Broadleaf Vegetation',
    'Annual Grass Vegetation', 'Non-Vegetated Lands', 'Urban and Built-up Lands',
]

def land_cover_percentages(location: ee.Geometry, year: int) -> Dict[str, Any]:
    """Annual land cover percentages from MODIS MCD12Q1 LC_Type4."""
    start_date, end_date = f"{year}-01-01", f"{year}-12-31"

    lc = (ee.ImageCollection('MODIS/061/MCD12Q1')
          .filterDate(start_date, end_date).mean().select('LC_Type4').clip(location))

    area_img = ee.Image.pixelArea().addBands(lc)
    grouped = area_img.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName='class'),
        geometry=location, scale=500, bestEffort=True, maxPixels=1e13,
    )

    groups = (grouped.get('groups') or ee.List([])).getInfo()
    class_area_map = {i: 0 for i in range(len(_LC_CLASS_NAMES))}
    total_area = 0.0

    for g in groups or []:
        cls, area = g.get('class'), g.get('sum', 0) or 0
        if isinstance(cls, int) and cls in class_area_map:
            class_area_map[cls] = area
            total_area += area

    if total_area <= 0:
        classes = {name: None for name in _LC_CLASS_NAMES}
    else:
        classes = {_LC_CLASS_NAMES[i]: (class_area_map[i] / total_area * 100.0) for i in range(len(_LC_CLASS_NAMES))}

    return {"year": year, "classes": classes}

# ---------------------------------------------------------------------------
# Hansen Forest Loss
# ---------------------------------------------------------------------------

def forest_loss_percentage(location: ee.Geometry, year: int) -> Dict[str, Any]:
    """Percent of region area with forest loss using Hansen GFC."""
    code = year - 2000
    if not (1 <= code <= 23):
        return {"year": year, "percent": None}

    dataset = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
    loss_mask = dataset.select('lossyear').eq(code)

    loss_area = (ee.Image.pixelArea().updateMask(loss_mask)
                 .reduceRegion(ee.Reducer.sum(), location, 30, maxPixels=1e13).get('area'))
    region_area = location.area()

    percent = ee.Number(loss_area).divide(region_area).multiply(100).getInfo()
    return {"year": year, "percent": percent}

# ---------------------------------------------------------------------------
# HTTP View
# ---------------------------------------------------------------------------

@method_decorator(csrf_exempt, name="dispatch")
class PrecipitationView(View):
    """
    POST /lepto/precipitation/
    Body: {"version": "2021", "region_id": "DE123", "year_month": "2021-02"}
    Response: Metrics + leptospirosis risk prediction.
    """
    def post(self, request) -> JsonResponse:
        try:
            ensure_ee_initialized()
            data = json.loads(request.body or "{}")

            version = data.get("version")
            region_id = data.get("region_id")
            year_month = data.get("year_month")
            start_date = data.get("start_date")
            end_date = data.get("end_date")

            # Validate required params
            if not version or not region_id:
                return JsonResponse({"error": "Missing 'version' or 'region_id'"}, status=400)

            if year_month:
                try:
                    dt = datetime.strptime(year_month, "%Y-%m")
                    start_date, end_date = month_window(dt.year, dt.month)
                except ValueError:
                    return JsonResponse({"error": "Invalid 'year_month'. Use YYYY-MM."}, status=400)

            if not start_date or not end_date:
                return JsonResponse({"error": "Missing 'start_date' and 'end_date' or 'year_month'"}, status=400)

            # Validate dates
            for d in (start_date, end_date):
                try:
                    datetime.strptime(d, "%Y-%m-%d")
                except ValueError:
                    return JsonResponse({"error": f"Invalid date: {d}. Use YYYY-MM-DD."}, status=400)

            # Load GeoJSON
            geojson_path = Path(settings.BASE_DIR).parent / "frontend" / "public" / f"nuts3_{version}.geojson"
            if not geojson_path.exists():
                return JsonResponse({"error": f"GeoJSON not found for version {version}"}, status=404)

            with open(geojson_path, "r", encoding="utf-8") as f:
                geo = json.load(f)

            feature = next(
                (feat for feat in geo.get("features", []) if feat.get("properties", {}).get("NUTS_ID") == region_id),
                None,
            )
            if not feature:
                return JsonResponse({"error": f"Region {region_id} not found in version {version}"}, status=404)

            geom = ee.Geometry(feature["geometry"])
            ym = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m")
            prev_start, prev_end = get_prev_month_window(ym)
            three_start, three_end = get_three_months_prior_start_end(ym)

            logger.info("Processing %s (%s): %s to %s", region_id, version, start_date, end_date)

            # Compute metrics
            res_precip = prec(geom, start_date, end_date)
            res_prev_precip = prec(geom, prev_start, prev_end)
            res_three_precip = prec(geom, three_start, three_end)
            res_temp = temp(geom, start_date, end_date)
            res_soil_moist = soil_moisture(geom, start_date, end_date)
            res_soil_temp = soil_temp(geom, start_date, end_date)
            res_ndwi = ndwi(geom, start_date, end_date)
            res_ndvi = ndvi(geom, start_date, end_date)

            year = int(start_date[:4])
            res_lc = land_cover_percentages(geom, year)
            res_loss = forest_loss_percentage(geom, year)

            # Fetch population density
            population_density = get_population_density(region_id, year)

            # Fetch GDP
            gdp = get_gdp(region_id, year)

            # Fetch employment
            employment = get_employment(region_id, year)

            # Fetch bovine population
            bovine_population = get_bovine(region_id, year, animals="A2000")

            # Prepare features dict for prediction
            features = {
                'Mean Temperature': res_temp.get('monthly_mean'),
                'Temperature Range': res_temp.get('range_hourly_mean'),
                'Mean Precipitation of Previous Month': res_prev_precip.get('monthly_mean'),
                'Maximum Precipitation of Previous Month': res_prev_precip.get('max_hourly_mean'),
                'Minimum Precipitation of Previous Month': res_prev_precip.get('min_hourly_mean'),
                'Mean Precipitation of Three Months Prior': res_three_precip.get('monthly_mean'),
                'Maximum Precipitation of Three Months Prior': res_three_precip.get('max_hourly_mean'),
                'Minimum Precipitation of Three Months Prior': res_three_precip.get('min_hourly_mean'),
                'Mean Precipitation': res_precip.get('monthly_mean'),
                'Maximum Precipitation': res_precip.get('max_hourly_mean'),
                'Minimum Precipitation': res_precip.get('min_hourly_mean'),
                'Soil Temperature Range': res_soil_temp.get('range_hourly_mean'),
                'Soil Moisture': res_soil_moist.get('monthly_mean'),
                'Maximum Soil Moisture': res_soil_moist.get('max_hourly_mean'),
                'Soil Moisture Range': res_soil_moist.get('range_hourly_mean'),
                'NDVI': res_ndvi.get('monthly_mean'),
                'NDWI': res_ndwi.get('monthly_mean'),
                'Water Bodies': res_lc.get('classes', {}).get('Water Bodies'),
                'Evergreen Needleleaf Vegetation': res_lc.get('classes', {}).get('Evergreen Needleleaf Vegetation'),
                'Evergreen Broadleaf Vegetation': res_lc.get('classes', {}).get('Evergreen Broadleaf Vegetation'),
                'Deciduous Needleleaf Vegetation': res_lc.get('classes', {}).get('Deciduous Needleleaf Vegetation'),
                'Deciduous Broadleaf Vegetation': res_lc.get('classes', {}).get('Deciduous Broadleaf Vegetation'),
                'Annual Broadleaf Vegetation': res_lc.get('classes', {}).get('Annual Broadleaf Vegetation'),
                'Annual Grass Vegetation': res_lc.get('classes', {}).get('Annual Grass Vegetation'),
                'Non-Vegetated Lands': res_lc.get('classes', {}).get('Non-Vegetated Lands'),
                'Urban and Built-up Lands': res_lc.get('classes', {}).get('Urban and Built-up Lands'),
                'Deforestation': res_loss.get('percent'),
                'GDP': gdp,
                'Employment': employment,
                'Population Density': population_density,
                'Bovine Population': bovine_population,
            }

            # Predict leptospirosis risk
            lepto_risk = predict_lepto_risk(features)

            payload = {
                "region_id": region_id,
                "version": version,
                "start_date": start_date,
                "end_date": end_date,
                "precipitation": res_precip,
                "previous_precipitation": {**res_prev_precip, "start_date": prev_start, "end_date": prev_end},
                "three_months_prior_precipitation": {**res_three_precip, "start_date": three_start, "end_date": three_end},
                "temperature": res_temp,
                "soil_moisture": res_soil_moist,
                "soil_temperature": res_soil_temp,
                "ndwi": res_ndwi,
                "ndvi": res_ndvi,
                "land_cover_percentages": res_lc,
                "forest_loss_percentage": res_loss,
                "population_density": population_density,
                "gdp": gdp,
                "employment": employment,
                "bovine_population": bovine_population,
                "predicted_probability": lepto_risk,
            }
            return JsonResponse(payload)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON body"}, status=400)
        except Exception as e:
            logger.error("API error: %s", str(e), exc_info=True)
            return JsonResponse({"error": "Internal server error"}, status=500)