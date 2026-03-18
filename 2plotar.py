#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Plot IFs from ifs-brasil.csv + Stations on a Brazil Map
Author: Heder Dorneles Soares
Institution: Instituto Nacional de Pesquisas Espaciais (INPE)
Laboratory: Laboratório de Inteligência Artificial para Aplicações AeroEspaciais e Ambientais (LIAREA)
Description:
    Reads Instituto Federal campus/unit coordinates from ifs-brasil.csv and station coordinates from
    stations.csv, caches the Natural Earth admin_1 borders locally, and plots both layers over a
    Brazil map using different markers and colors.

Created: 2026-03-06
"""

import os
import zipfile
import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
IFS_CSV = "ifs-brasil.csv"
STATIONS_CSV = "stations.csv"          # format: name,city,state,lat,lon
CACHE_DIR = "./ne_admin1_cache"
OUT_PNG = "map_ifs_stations.png"

NE_ZIP = "ne_10m_admin_1_states_provinces.zip"
NE_SHP = "ne_10m_admin_1_states_provinces.shp"
NE_URL = f"https://naturalearth.s3.amazonaws.com/10m_cultural/{NE_ZIP}"

# Markers / colors
IFS_MARKER = "o"
IFS_COLOR = "blue"

STATIONS_MARKER = "^"   # triangle; use "s" for square if you prefer
STATIONS_COLOR = "red"

# =========================
# Helpers
# =========================
def pick_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def to_float_series(s):
    return (
        s.astype(str)
         .str.strip()
         .str.replace(",", ".", regex=False)
         .replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
         .pipe(pd.to_numeric, errors="coerce")
    )

def read_csv_robust(path):
    for sep in [";", ",", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8", low_memory=False)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")

def ensure_naturalearth_admin1(cache_dir=CACHE_DIR, url=NE_URL, zip_name=NE_ZIP, shp_name=NE_SHP):
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, zip_name)
    shp_path = os.path.join(cache_dir, shp_name)

    # If shapefile exists, skip download/extract
    if os.path.exists(shp_path):
        return shp_path

    # If zip doesn't exist, download
    if not os.path.exists(zip_path):
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    # Extract only if required pieces are missing
    with zipfile.ZipFile(zip_path, "r") as z:
        needed_exts = [".shp", ".shx", ".dbf", ".prj"]
        present = set(os.listdir(cache_dir))
        base = os.path.splitext(shp_name)[0]
        if not all((base + ext) in present for ext in needed_exts):
            z.extractall(cache_dir)

    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Could not find {shp_path} after extraction.")
    return shp_path

def clean_points_df(df, lat_col, lon_col):
    df[lat_col] = to_float_series(df[lat_col])
    df[lon_col] = to_float_series(df[lon_col])
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    df = df[~((df[lat_col] == 0) & (df[lon_col] == 0))].copy()
    return df

# =========================
# 1) IFs points
# =========================
df_ifs = read_csv_robust(IFS_CSV)

ifs_lat = pick_col(df_ifs, ["LATITUDE", "latitude", "lat", "y"])
ifs_lon = pick_col(df_ifs, ["LONGITUDE", "longitude", "lon", "lng", "x"])

if not ifs_lat or not ifs_lon:
    raise SystemExit(
        "Could not find latitude/longitude columns in IFs CSV.\n"
        f"Columns found: {list(df_ifs.columns)}"
    )

df_ifs = clean_points_df(df_ifs, ifs_lat, ifs_lon)
df_ifs = df_ifs.drop_duplicates(subset=[ifs_lat, ifs_lon]).copy()

gdf_ifs = gpd.GeoDataFrame(
    df_ifs,
    geometry=gpd.points_from_xy(df_ifs[ifs_lon], df_ifs[ifs_lat]),
    crs="EPSG:4326"
)

print(f"IFs CSV: {IFS_CSV}")
print(f"Using columns: lat={ifs_lat} lon={ifs_lon}")
print(f"IF points after cleaning/dedup: {len(gdf_ifs)}")

# =========================
# 2) Stations (name,city,state,lat,lon)
# =========================
df_st = pd.read_csv(STATIONS_CSV, encoding="utf-8")

# Ensure required columns exist (case-insensitive)
required = ["name", "city", "state", "lat", "lon"]
lower_cols = {c.lower(): c for c in df_st.columns}
missing = [c for c in required if c not in lower_cols]
if missing:
    raise SystemExit(
        f"Stations CSV is missing columns: {missing}\n"
        f"Columns found: {list(df_st.columns)}"
    )

st_lat = lower_cols["lat"]
st_lon = lower_cols["lon"]

df_st = clean_points_df(df_st, st_lat, st_lon)
df_st = df_st.drop_duplicates(subset=[st_lat, st_lon]).copy()

gdf_st = gpd.GeoDataFrame(
    df_st,
    geometry=gpd.points_from_xy(df_st[st_lon], df_st[st_lat]),
    crs="EPSG:4326"
)

print(f"Stations CSV: {STATIONS_CSV}")
print(f"Stations points after cleaning/dedup: {len(gdf_st)}")

# =========================
# 3) Brazil admin_1 borders (Natural Earth) with cache
# =========================
shp_path = ensure_naturalearth_admin1()
states = gpd.read_file(shp_path).to_crs("EPSG:4326")

br_states = None
if "admin" in states.columns:
    br_states = states[states["admin"] == "Brazil"].copy()
if (br_states is None or len(br_states) == 0) and "adm0_a3" in states.columns:
    br_states = states[states["adm0_a3"] == "BRA"].copy()
if (br_states is None or len(br_states) == 0) and "iso_a2" in states.columns:
    br_states = states[states["iso_a2"] == "BR"].copy()
if br_states is None or len(br_states) == 0:
    raise SystemExit("Could not filter Brazil states from the Natural Earth shapefile.")

# =========================
# 4) Plot (English labels)
# =========================
fig, ax = plt.subplots(figsize=(10, 10))

# State borders
br_states.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.6)

# IFs (circles)
gdf_ifs.plot(
    ax=ax,
    markersize=10,
    alpha=0.7,
    marker=IFS_MARKER,
    color=IFS_COLOR,
    label="Federal Institutes (from CSV)"
)

# Stations (triangles or squares)
gdf_st.plot(
    ax=ax,
    markersize=60,
    alpha=0.9,
    marker=STATIONS_MARKER,
    color=STATIONS_COLOR,
    label="Stations (from CSV)"
)

# Frame to Brazil bounds
minx, miny, maxx, maxy = br_states.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

ax.set_title("Brazil — Federal Institutes and Stations")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(loc="lower left")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
print(f"Map saved to: {OUT_PNG}")
plt.show()