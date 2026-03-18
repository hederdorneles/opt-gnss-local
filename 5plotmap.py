#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nome: Heder Dorneles Soares
Instituição: INPE
Laboratório: LIAREA (Laboratório de Inteligência Artificial para Aplicações AeroEspaciais e Ambientais)

Plot Brazil map with:
- IF candidates (ifs-brasil.csv)
- Fixed stations (inct_active_stations.csv)
- Selected new IF sites (out/selected_sites.csv from the optimizer)

Output: out/map_solution.png (by default)
"""

import os
import zipfile
import argparse

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import requests


# =========================
# Natural Earth (admin_1) cache
# =========================
NE_ZIP = "ne_10m_admin_1_states_provinces.zip"
NE_SHP = "ne_10m_admin_1_states_provinces.shp"
NE_URL = f"https://naturalearth.s3.amazonaws.com/10m_cultural/{NE_ZIP}"


def pick_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def to_float_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.replace(",", ".", regex=False)
         .replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
         .pipe(pd.to_numeric, errors="coerce")
    )


def read_csv_robust(path: str) -> pd.DataFrame:
    for sep in [";", ",", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8", low_memory=False)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8", low_memory=False)


def ensure_naturalearth_admin1(cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, NE_ZIP)
    shp_path = os.path.join(cache_dir, NE_SHP)

    if os.path.exists(shp_path):
        return shp_path

    if not os.path.exists(zip_path):
        r = requests.get(NE_URL, timeout=60, stream=True)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as z:
        needed_exts = [".shp", ".shx", ".dbf", ".prj"]
        base = os.path.splitext(NE_SHP)[0]
        present = set(os.listdir(cache_dir))
        if not all((base + ext) in present for ext in needed_exts):
            z.extractall(cache_dir)

    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Could not find {shp_path} after extraction.")

    return shp_path


def clean_points_df(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    df = df.copy()
    df[lat_col] = to_float_series(df[lat_col])
    df[lon_col] = to_float_series(df[lon_col])
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    df = df[~((df[lat_col] == 0) & (df[lon_col] == 0))].copy()
    df = df.drop_duplicates(subset=[lat_col, lon_col]).copy()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ifs_csv", default="ifs-brasil.csv")
    ap.add_argument("--stations_csv", default="inct_active_stations.csv")
    ap.add_argument("--selected_csv", default=os.path.join("out", "selected_sites.csv"))
    ap.add_argument("--cache_dir", default="./ne_admin1_cache")
    ap.add_argument("--out_png", default=os.path.join("out", "map_solution.png"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)

    # =========================
    # 1) IF candidates
    # =========================
    df_ifs = read_csv_robust(args.ifs_csv)
    ifs_lat = pick_col(df_ifs, ["Latitude", "LATITUDE", "lat", "y"])
    ifs_lon = pick_col(df_ifs, ["Longitude", "LONGITUDE", "lon", "lng", "x"])
    if not ifs_lat or not ifs_lon:
        raise SystemExit(f"Could not find lat/lon columns in IFs CSV. Columns: {list(df_ifs.columns)}")
    df_ifs = clean_points_df(df_ifs, ifs_lat, ifs_lon)

    gdf_ifs = gpd.GeoDataFrame(
        df_ifs,
        geometry=gpd.points_from_xy(df_ifs[ifs_lon], df_ifs[ifs_lat]),
        crs="EPSG:4326",
    )

    # =========================
    # 2) Fixed stations
    # =========================
    df_st = read_csv_robust(args.stations_csv)
    st_lat = pick_col(df_st, ["lat", "Latitude", "LAT"])
    st_lon = pick_col(df_st, ["lon", "Longitude", "LON"])
    if not st_lat or not st_lon:
        raise SystemExit(f"Could not find lat/lon columns in stations CSV. Columns: {list(df_st.columns)}")
    df_st = clean_points_df(df_st, st_lat, st_lon)

    gdf_st = gpd.GeoDataFrame(
        df_st,
        geometry=gpd.points_from_xy(df_st[st_lon], df_st[st_lat]),
        crs="EPSG:4326",
    )

    # =========================
    # 3) Selected new sites (from optimizer output)
    # =========================
    if not os.path.exists(args.selected_csv):
        raise SystemExit(f"selected_sites.csv not found: {args.selected_csv}")

    df_sel = read_csv_robust(args.selected_csv)
    sel_lat = pick_col(df_sel, ["Latitude", "lat", "LATITUDE"])
    sel_lon = pick_col(df_sel, ["Longitude", "lon", "LONGITUDE"])
    if not sel_lat or not sel_lon:
        raise SystemExit(f"Could not find lat/lon columns in selected CSV. Columns: {list(df_sel.columns)}")

    stype = pick_col(df_sel, ["site_type", "Site_Type", "type"])
    if not stype:
        raise SystemExit(f"Could not find site_type column in selected CSV. Columns: {list(df_sel.columns)}")

    df_sel[stype] = df_sel[stype].astype(str).str.strip().str.lower()
    df_new = df_sel[df_sel[stype].isin(["selected_new", "selected"])].copy()
    df_new = clean_points_df(df_new, sel_lat, sel_lon)

    gdf_new = gpd.GeoDataFrame(
        df_new,
        geometry=gpd.points_from_xy(df_new[sel_lon], df_new[sel_lat]),
        crs="EPSG:4326",
    )

    # =========================
    # 4) Brazil admin_1 borders (Natural Earth)
    # =========================
    shp_path = ensure_naturalearth_admin1(args.cache_dir)
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
    # 5) Plot
    # =========================
    fig, ax = plt.subplots(figsize=(10, 10))

    br_states.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.6)

    # IF candidates
    gdf_ifs.plot(
        ax=ax,
        markersize=8,
        alpha=0.35,
        marker="o",
        color="tab:blue",
        label="IF candidates",
    )

    # Fixed stations
    gdf_st.plot(
        ax=ax,
        markersize=70,
        alpha=0.9,
        marker="^",
        color="tab:red",
        label="Fixed stations",
    )

    # Selected new IF sites
    if len(gdf_new) > 0:
        gdf_new.plot(
            ax=ax,
            markersize=140,
            alpha=0.95,
            marker="*",
            color="gold",
            edgecolor="black",
            linewidth=0.6,
            label="Selected new sites",
        )

    minx, miny, maxx, maxy = br_states.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    ax.set_title("Brazil — IF candidates, fixed stations, and selected new sites")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"Map saved to: {args.out_png}")
    plt.show()


if __name__ == "__main__":
    main()