#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Plot IFs from ifs-brasil.csv + Stations on a Brazil Map (with inset zoom)
Author: Heder Dorneles Soares
Institution: Instituto Nacional de Pesquisas Espaciais (INPE)
Laboratory: Laboratório de Inteligência Artificial para Aplicações AeroEspaciais e Ambientais (LIAREA)
Description:
    Reads Instituto Federal campus/unit coordinates from ifs-brasil.csv and station coordinates from
    stations.csv (name,city,state,lat,lon), caches the Natural Earth admin_1 borders locally, and
    plots both layers over a Brazil map using different markers and colors. Adds a smaller inset
    (zoom) around Presidente Prudente in the lower-right corner to separate nearby stations.

Created: 2026-03-06
"""
import os
import zipfile
import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# =========================
# Config
# =========================
IFS_CSV = "ifs-brasil.csv"
STATIONS_CSV = "stations.csv"          # format: name,city,state,lat,lon
CACHE_DIR = "./ne_admin1_cache"
OUT_PNG = "map_ifs_stations_inset.png"

NE_ZIP = "ne_10m_admin_1_states_provinces.zip"
NE_SHP = "ne_10m_admin_1_states_provinces.shp"
NE_URL = f"https://naturalearth.s3.amazonaws.com/10m_cultural/{NE_ZIP}"

# Markers / colors
IFS_MARKER = "o"
IFS_COLOR = "blue"
IFS_SIZE = 10

STATIONS_MARKER = "^"   # triangle; use "s" for square
STATIONS_COLOR = "red"
STATIONS_SIZE = 60

# Inset settings (Presidente Prudente region)
INSET_ENABLE = True
INSET_PADDING_DEG = 0.25
INSET_LOC = "lower right"   # bottom-right
INSET_W = "28%"             # smaller
INSET_H = "28%"             # smaller
INSET_CITY = "Presidente Prudente"

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

def filter_brazil_admin1(states: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    br_states = None
    if "admin" in states.columns:
        br_states = states[states["admin"] == "Brazil"].copy()
    if (br_states is None or len(br_states) == 0) and "adm0_a3" in states.columns:
        br_states = states[states["adm0_a3"] == "BRA"].copy()
    if (br_states is None or len(br_states) == 0) and "iso_a2" in states.columns:
        br_states = states[states["iso_a2"] == "BR"].copy()
    if br_states is None or len(br_states) == 0:
        raise SystemExit("Could not filter Brazil states from the Natural Earth shapefile.")
    return br_states

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
# 2) Stations points (schema: name,city,state,lat,lon)
# =========================
df_st = pd.read_csv(STATIONS_CSV, encoding="utf-8")

lower_cols = {c.lower(): c for c in df_st.columns}
required = ["name", "city", "state", "lat", "lon"]
missing = [c for c in required if c not in lower_cols]
if missing:
    raise SystemExit(
        f"Stations CSV missing columns: {missing}\n"
        f"Columns found: {list(df_st.columns)}"
    )

name_col = lower_cols["name"]
city_col = lower_cols["city"]
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
# 3) Brazil borders (Natural Earth) with cache
# =========================
shp_path = ensure_naturalearth_admin1()
states = gpd.read_file(shp_path).to_crs("EPSG:4326")
br_states = filter_brazil_admin1(states)

# =========================
# 4) Main plot
# =========================
fig, ax = plt.subplots(figsize=(10, 10))

br_states.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.6)

gdf_ifs.plot(
    ax=ax,
    markersize=IFS_SIZE,
    alpha=0.7,
    marker=IFS_MARKER,
    color=IFS_COLOR,
    label="Federal Institutes"
)

gdf_st.plot(
    ax=ax,
    markersize=STATIONS_SIZE,
    alpha=0.9,
    marker=STATIONS_MARKER,
    color=STATIONS_COLOR,
    label="Stations"
)

minx, miny, maxx, maxy = br_states.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

ax.set_title("Brazil — Federal Institutes and Stations")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(loc="lower left")

# =========================
# 5) Inset zoom (Presidente Prudente cluster)
# =========================
if INSET_ENABLE:
    gdf_pp = gdf_st[gdf_st[city_col].astype(str).str.strip().str.lower() == INSET_CITY.lower()].copy()
    if len(gdf_pp) < 2:
        gdf_pp = gdf_st[gdf_st[name_col].astype(str).str.upper().str.startswith("PRU")].copy()

    if len(gdf_pp) >= 2:
        minx2, miny2, maxx2, maxy2 = gdf_pp.total_bounds
        minx2 -= INSET_PADDING_DEG
        miny2 -= INSET_PADDING_DEG
        maxx2 += INSET_PADDING_DEG
        maxy2 += INSET_PADDING_DEG

        axins = inset_axes(ax, width=INSET_W, height=INSET_H, loc=INSET_LOC)

        br_states.plot(ax=axins, facecolor="none", edgecolor="black", linewidth=0.6)
        gdf_ifs.plot(ax=axins, markersize=IFS_SIZE, alpha=0.5, marker=IFS_MARKER, color=IFS_COLOR)
        gdf_st.plot(ax=axins, markersize=STATIONS_SIZE, alpha=0.95, marker=STATIONS_MARKER, color=STATIONS_COLOR)

        axins.set_xlim(minx2, maxx2)
        axins.set_ylim(miny2, maxy2)
        axins.set_title("Zoom: Presidente Prudente", fontsize=10)
        axins.set_xticks([])
        axins.set_yticks([])

        # Connector lines better suited for a lower-right inset
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="black", lw=0.8)
    else:
        print("Inset skipped: not enough stations found for Presidente Prudente cluster.")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
print(f"Map saved to: {OUT_PNG}")
plt.show()