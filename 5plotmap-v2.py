#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nome: Heder Dorneles Soares
Instituição: INPE
Laboratório: LIAREA (Laboratório de Inteligência Artificial para Aplicações AeroEspaciais e Ambientais)

Relocation-only greedy solver (MCLP-style) using EXACTLY the same number of active stations (N_ativas),
redistributed among candidate locations (IF campuses + current station locations).

Além de gerar os CSVs/metrics, este script TENTA plotar automaticamente um mapa do Brasil com
APENAS as novas localidades selecionadas (rede otimizada).

Outputs
-------
- out/selected_sites.csv
- out/baseline_stations.csv
- out/metrics.json
- out/map_selected_only.png   (raster)
- out/map_selected_only.pdf   (vetorial, recomendado para LaTeX)

Dependencies
------------
Required: numpy, pandas
Optional (for Brazil clipping): geopandas, shapely
Optional (for map): geopandas, matplotlib  (e dependências do geopandas: fiona/pyproj/shapely)
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


# ----------------------------
# Utilities
# ----------------------------
def read_csv_robust(path: str) -> pd.DataFrame:
    """Try comma, then semicolon; handle common encoding issues."""
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")


def to_float_series(s: pd.Series) -> pd.Series:
    """Convert numeric strings possibly using comma decimal separator."""
    s2 = s.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s2, errors="coerce")


def haversine_matrix_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Pairwise haversine distance (km). Returns matrix (len(lat1), len(lat2))."""
    lat1r = np.radians(lat1).reshape(-1, 1)
    lon1r = np.radians(lon1).reshape(-1, 1)
    lat2r = np.radians(lat2).reshape(1, -1)
    lon2r = np.radians(lon2).reshape(1, -1)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return EARTH_RADIUS_KM * c


def ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _fix_latlon_and_dedupe_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure unique columns; normalize lat/lon into Latitude/Longitude safely."""
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if "Latitude" in df.columns and "lat" in df.columns:
        df["Latitude"] = df["Latitude"].fillna(df["lat"])
        df.drop(columns=["lat"], inplace=True)
    elif "Latitude" not in df.columns and "lat" in df.columns:
        df.rename(columns={"lat": "Latitude"}, inplace=True)

    if "Longitude" in df.columns and "lon" in df.columns:
        df["Longitude"] = df["Longitude"].fillna(df["lon"])
        df.drop(columns=["lon"], inplace=True)
    elif "Longitude" not in df.columns and "lon" in df.columns:
        df.rename(columns={"lon": "Longitude"}, inplace=True)

    df.columns = [str(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    return df


# ----------------------------
# Optional: Brazil polygon clipping
# ----------------------------
def try_load_brazil_polygon(cache_dir: str) -> Optional["object"]:
    """Return Brazil geometry from Natural Earth admin_0, or None if unavailable."""
    try:
        import geopandas as gpd  # type: ignore
    except Exception:
        return None

    ensure_dir(cache_dir)
    zip_path = os.path.join(cache_dir, "ne_10m_admin_0_countries.zip")

    if not os.path.exists(zip_path):
        url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
        try:
            import urllib.request
            urllib.request.urlretrieve(url, zip_path)
        except Exception:
            return None

    try:
        world = gpd.read_file(f"zip://{zip_path}")
    except Exception:
        return None

    for col in ["ADM0_A3", "ISO_A3", "adm0_a3"]:
        if col in world.columns:
            bra = world[world[col].astype(str).str.upper() == "BRA"]
            if len(bra) == 1:
                return bra.geometry.values[0]

    for col in ["NAME", "ADMIN", "name", "admin"]:
        if col in world.columns:
            bra = world[world[col].astype(str).str.lower() == "brazil"]
            if len(bra) == 1:
                return bra.geometry.values[0]

    return None


def build_demand_grid(
    bounds: Tuple[float, float, float, float],
    grid_deg: float,
    brazil_poly: Optional["object"] = None,
) -> pd.DataFrame:
    """Regular lat/lon grid within bounds; optionally clip to Brazil polygon."""
    min_lon, min_lat, max_lon, max_lat = bounds

    lats = np.arange(min_lat, max_lat + 1e-9, grid_deg)
    lons = np.arange(min_lon, max_lon + 1e-9, grid_deg)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    pts = pd.DataFrame({"lat": lat_grid.ravel().astype(float), "lon": lon_grid.ravel().astype(float)})

    if brazil_poly is None:
        return pts

    try:
        from shapely.geometry import Point  # type: ignore
        from shapely.prepared import prep  # type: ignore
    except Exception:
        return pts

    prepared = prep(brazil_poly)
    mask = np.array([prepared.contains(Point(xy)) for xy in zip(pts["lon"], pts["lat"])], dtype=bool)
    return pts.loc[mask].reset_index(drop=True)


# ----------------------------
# Solver
# ----------------------------
@dataclass
class Solution:
    selected_candidate_idx: list
    coverage_baseline: float
    coverage_optimized: float
    covered_weight_baseline: float
    covered_weight_optimized: float
    n_demand: int


def greedy_mclp_exact_p(
    cover_cand: np.ndarray,      # (nD, nC) bool
    weights: np.ndarray,         # (nD,) float
    p_select: int,
    min_sep_km: float,
    cand_lat: np.ndarray,
    cand_lon: np.ndarray,
) -> list[int]:
    """
    Greedy maximum coverage selecting EXACTLY p_select candidates (or fail).
    If marginal gain becomes 0, it still selects the best available to reach p_select.
    """
    nD, nC = cover_cand.shape
    selected: list[int] = []
    covered = np.zeros(nD, dtype=bool)

    cover_uint8 = cover_cand.astype(np.uint8)

    cand_cand_dist = None
    min_dist_to_selected = np.full(nC, np.inf, dtype=float)
    if min_sep_km > 0:
        cand_cand_dist = haversine_matrix_km(cand_lat, cand_lon, cand_lat, cand_lon)

    for _ in range(max(0, p_select)):
        uncovered_w = weights.copy()
        uncovered_w[covered] = 0.0

        gains = (cover_uint8.T @ uncovered_w).astype(float)

        if selected:
            gains[np.array(selected, dtype=int)] = -1.0

        if min_sep_km > 0 and cand_cand_dist is not None:
            too_close = min_dist_to_selected < min_sep_km
            gains[too_close] = -1.0

        j = int(np.argmax(gains))
        if gains[j] < 0:
            break

        selected.append(j)
        covered |= cover_cand[:, j]

        if min_sep_km > 0 and cand_cand_dist is not None:
            min_dist_to_selected = np.minimum(min_dist_to_selected, cand_cand_dist[:, j])

    return selected


def compute_covered_by_sites(
    demand_lat: np.ndarray,
    demand_lon: np.ndarray,
    site_lat: np.ndarray,
    site_lon: np.ndarray,
    radius_km: float,
    chunk_sites: int = 64,
) -> np.ndarray:
    """Demand covered by any site within radius_km."""
    nD = len(demand_lat)
    covered = np.zeros(nD, dtype=bool)
    if len(site_lat) == 0:
        return covered

    for s0 in range(0, len(site_lat), chunk_sites):
        s1 = min(len(site_lat), s0 + chunk_sites)
        d = haversine_matrix_km(demand_lat, demand_lon, site_lat[s0:s1], site_lon[s0:s1])
        covered |= (d <= radius_km).any(axis=1)
        if covered.all():
            break
    return covered


def compute_nearest_distance_km(
    demand_lat: np.ndarray,
    demand_lon: np.ndarray,
    site_lat: np.ndarray,
    site_lon: np.ndarray,
    chunk_sites: int = 64,
) -> np.ndarray:
    """Nearest distance (km) from each demand point to any site."""
    nD = len(demand_lat)
    nearest = np.full(nD, np.inf, dtype=float)
    if len(site_lat) == 0:
        return nearest

    for s0 in range(0, len(site_lat), chunk_sites):
        s1 = min(len(site_lat), s0 + chunk_sites)
        d = haversine_matrix_km(demand_lat, demand_lon, site_lat[s0:s1], site_lon[s0:s1])
        nearest = np.minimum(nearest, d.min(axis=1))
    return nearest


# ----------------------------
# Map plotting (selected only)
# ----------------------------
NE_ADMIN1_ZIP = "ne_10m_admin_1_states_provinces.zip"
NE_ADMIN1_SHP = "ne_10m_admin_1_states_provinces.shp"
NE_ADMIN1_URL = f"https://naturalearth.s3.amazonaws.com/10m_cultural/{NE_ADMIN1_ZIP}"


def ensure_naturalearth_admin1(cache_dir: str) -> str:
    ensure_dir(cache_dir)
    zip_path = os.path.join(cache_dir, NE_ADMIN1_ZIP)
    shp_path = os.path.join(cache_dir, NE_ADMIN1_SHP)

    if os.path.exists(shp_path):
        return shp_path

    if not os.path.exists(zip_path):
        import urllib.request
        urllib.request.urlretrieve(NE_ADMIN1_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(cache_dir)

    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Could not find {shp_path} after extraction.")
    return shp_path


def plot_selected_only_map(
    selected_out_csv: str,
    cache_dir: str,
    out_png: str,
    out_pdf: str,
    png_dpi: int = 300,
) -> None:
    """
    Plot Brazil outline + ONLY selected new locations (from selected_sites.csv).
    Saves both PNG (raster) and PDF (vector).
    """
    import geopandas as gpd  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    if not os.path.exists(selected_out_csv):
        raise FileNotFoundError(f"selected_sites.csv not found: {selected_out_csv}")

    df_sel = read_csv_robust(selected_out_csv)
    lat_col = "Latitude" if "Latitude" in df_sel.columns else ("lat" if "lat" in df_sel.columns else None)
    lon_col = "Longitude" if "Longitude" in df_sel.columns else ("lon" if "lon" in df_sel.columns else None)
    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not find Latitude/Longitude in {selected_out_csv}. Columns: {list(df_sel.columns)}")

    df_sel[lat_col] = to_float_series(df_sel[lat_col])
    df_sel[lon_col] = to_float_series(df_sel[lon_col])
    df_sel = df_sel.dropna(subset=[lat_col, lon_col]).copy()
    if len(df_sel) == 0:
        raise ValueError("No selected points to plot (selected_sites.csv is empty after cleaning).")

    gdf_sel = gpd.GeoDataFrame(
        df_sel,
        geometry=gpd.points_from_xy(df_sel[lon_col], df_sel[lat_col]),
        crs="EPSG:4326",
    )

    shp_path = ensure_naturalearth_admin1(cache_dir)
    states = gpd.read_file(shp_path).to_crs("EPSG:4326")

    br_states = None
    if "admin" in states.columns:
        br_states = states[states["admin"] == "Brazil"].copy()
    if (br_states is None or len(br_states) == 0) and "adm0_a3" in states.columns:
        br_states = states[states["adm0_a3"] == "BRA"].copy()
    if (br_states is None or len(br_states) == 0) and "iso_a2" in states.columns:
        br_states = states[states["iso_a2"] == "BR"].copy()
    if br_states is None or len(br_states) == 0:
        raise RuntimeError("Could not filter Brazil states from Natural Earth admin_1.")

    ensure_dir(os.path.dirname(out_png) or ".")
    ensure_dir(os.path.dirname(out_pdf) or ".")

    fig, ax = plt.subplots(figsize=(10, 10))
    br_states.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.6)

    gdf_sel.plot(
        ax=ax,
        markersize=140,
        alpha=0.95,
        marker="*",
        edgecolor="black",
        linewidth=0.6,
        label="Selected stations",
    )

    minx, miny, maxx, maxy = br_states.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    ax.set_title("Brazil — Optimized station locations")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")                 # vector (LaTeX)
    plt.savefig(out_png, dpi=png_dpi, bbox_inches="tight")    # raster (optional)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Relocation-only greedy MCLP: select EXACTLY N_ativas among candidates.")
    ap.add_argument("--stations_csv", default="inct_active_stations.csv", help="Existing active stations CSV (baseline).")
    ap.add_argument("--ifs_csv", default="ifs-brasil.csv", help="Federal Institutes CSV (candidates).")
    ap.add_argument("--out_dir", default="out", help="Output directory.")
    ap.add_argument("--radius_km", type=float, default=250.0, help="Coverage radius r (km).")
    ap.add_argument("--grid_deg", type=float, default=0.5, help="Demand grid resolution in degrees.")
    ap.add_argument(
        "--p_total",
        type=int,
        default=None,
        help="Total stations to select (defaults to N_ativas from stations_csv).",
    )
    ap.add_argument(
        "--min_sep_km",
        type=float,
        default=0.0,
        help="Min separation between selected sites (km). Use 0 to allow all N_ativas.",
    )
    ap.add_argument(
        "--weight_mode",
        choices=["uniform", "inverse_distance_to_baseline"],
        default="uniform",
        help="Demand weighting scheme.",
    )
    ap.add_argument("--weight_alpha", type=float, default=1.0, help="Alpha for inverse-distance weighting.")
    ap.add_argument("--clip_brazil", action="store_true", help="Clip demand grid to Brazil polygon (Natural Earth download).")
    ap.add_argument("--cache_dir", default=".cache_geo", help="Cache directory for Natural Earth downloads.")
    ap.add_argument("--map_cache_dir", default="./ne_admin1_cache", help="Cache directory for Natural Earth admin_1.")
    ap.add_argument("--map_png_dpi", type=int, default=300, help="DPI for PNG map output.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(args.cache_dir)
    ensure_dir(args.map_cache_dir)

    # ----------------------------
    # Load CSVs
    # ----------------------------
    st_raw = read_csv_robust(args.stations_csv).copy()
    ifs = read_csv_robust(args.ifs_csv).copy()

    n_raw = int(len(st_raw))

    # Baseline stations lat/lon
    st = st_raw.copy()
    if "lat" in st.columns and "lon" in st.columns:
        st["lat"] = to_float_series(st["lat"])
        st["lon"] = to_float_series(st["lon"])
    else:
        found = False
        for latc in ["Latitude", "LAT", "Lat", "latitude"]:
            for lonc in ["Longitude", "LON", "Lon", "longitude"]:
                if latc in st.columns and lonc in st.columns:
                    st["lat"] = to_float_series(st[latc])
                    st["lon"] = to_float_series(st[lonc])
                    found = True
                    break
            if found:
                break

    st = st.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    n_ativas = int(len(st))

    # IF candidates lat/lon
    if "Latitude" in ifs.columns and "Longitude" in ifs.columns:
        ifs["lat"] = to_float_series(ifs["Latitude"])
        ifs["lon"] = to_float_series(ifs["Longitude"])
    else:
        raise ValueError("ifs-brasil.csv must contain Latitude and Longitude columns.")
    ifs = ifs.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # EXACTLY N_ativas by default
    p_total = n_ativas if args.p_total is None else int(args.p_total)

    print(f"[INFO] Quantidade de estações no arquivo (raw): {n_raw}")
    print(f"[INFO] Estações utilizáveis (com lat/lon): {n_ativas}")
    print(f"[INFO] Quantas estações serão utilizadas (p_total): {p_total}")

    # ----------------------------
    # Candidates C = IF campuses + current station locations
    # ----------------------------
    cand_ifs = ifs.copy()
    cand_ifs["candidate_source"] = "IF"
    cand_ifs["site_type"] = "candidate"

    cand_st = st.copy()
    cand_st["candidate_source"] = "EXISTING_STATION"
    cand_st["site_type"] = "candidate"

    cand = pd.concat([cand_ifs, cand_st], ignore_index=True, sort=False)

    # Drop exact coordinate duplicates
    cand["_lat_round"] = cand["lat"].round(7)
    cand["_lon_round"] = cand["lon"].round(7)
    cand = (
        cand.drop_duplicates(subset=["_lat_round", "_lon_round"])
        .drop(columns=["_lat_round", "_lon_round"])
        .reset_index(drop=True)
    )

    print(f"[INFO] Quantidade de locais candidatos (IFs + estações atuais, sem duplicados): {len(cand)}")

    if len(cand) < p_total:
        raise SystemExit(
            f"[ERRO] Número de candidatos ({len(cand)}) é menor que p_total ({p_total}). "
            "Isso pode ocorrer se muitos pontos forem duplicados."
        )

    baseline_lat = st["lat"].to_numpy(dtype=float)
    baseline_lon = st["lon"].to_numpy(dtype=float)
    cand_lat = cand["lat"].to_numpy(dtype=float)
    cand_lon = cand["lon"].to_numpy(dtype=float)

    # ----------------------------
    # Demand points D
    # ----------------------------
    all_lat = np.concatenate([baseline_lat, cand_lat]) if len(baseline_lat) else cand_lat
    all_lon = np.concatenate([baseline_lon, cand_lon]) if len(baseline_lon) else cand_lon

    if len(all_lat) == 0 or len(all_lon) == 0:
        bounds = (-74.0, -34.0, -34.0, 6.0)
    else:
        min_lat = float(np.nanmin(all_lat)) - 1.0
        max_lat = float(np.nanmax(all_lat)) + 1.0
        min_lon = float(np.nanmin(all_lon)) - 1.0
        max_lon = float(np.nanmax(all_lon)) + 1.0
        bounds = (max(-74.5, min_lon), max(-35.0, min_lat), min(-33.5, max_lon), min(6.5, max_lat))

    brazil_poly = try_load_brazil_polygon(args.cache_dir) if args.clip_brazil else None

    demand = build_demand_grid(bounds=bounds, grid_deg=args.grid_deg, brazil_poly=brazil_poly)
    dlat = demand["lat"].to_numpy(dtype=float)
    dlon = demand["lon"].to_numpy(dtype=float)
    nD = int(len(demand))

    # ----------------------------
    # Weights
    # ----------------------------
    weights = np.ones(nD, dtype=float)

    covered_baseline = compute_covered_by_sites(dlat, dlon, baseline_lat, baseline_lon, args.radius_km)

    if args.weight_mode == "inverse_distance_to_baseline":
        nearest_baseline = compute_nearest_distance_km(dlat, dlon, baseline_lat, baseline_lon)
        med = float(np.nanmedian(nearest_baseline[np.isfinite(nearest_baseline)])) if np.isfinite(nearest_baseline).any() else 1.0
        weights = 1.0 + args.weight_alpha * (nearest_baseline / max(med, 1e-6))

    total_weight = float(weights.sum())
    covered_weight_baseline = float(weights[covered_baseline].sum())
    coverage_baseline = covered_weight_baseline / total_weight if total_weight > 0 else 0.0

    # ----------------------------
    # Coverage matrix a_ij for candidates
    # ----------------------------
    dist_dc = haversine_matrix_km(dlat, dlon, cand_lat, cand_lon)
    cover_cand = dist_dc <= float(args.radius_km)

    # ----------------------------
    # Solve: select EXACTLY p_total sites among candidates
    # ----------------------------
    selected_idx = greedy_mclp_exact_p(
        cover_cand=cover_cand,
        weights=weights,
        p_select=p_total,
        min_sep_km=float(args.min_sep_km),
        cand_lat=cand_lat,
        cand_lon=cand_lon,
    )

    if len(selected_idx) != p_total:
        raise SystemExit(
            f"[ERRO] Foram selecionados {len(selected_idx)} locais, mas deveria ser {p_total}. "
            f"Isso geralmente ocorre por restrição de --min_sep_km muito alta. "
            f"Tente reduzir (ex.: --min_sep_km 0, 10, 20)."
        )

    print(f"[INFO] Total de localidades selecionadas: {len(selected_idx)}")

    # Optimized coverage
    covered_opt = cover_cand[:, selected_idx].any(axis=1)
    covered_weight_opt = float(weights[covered_opt].sum())
    coverage_opt = covered_weight_opt / total_weight if total_weight > 0 else 0.0

    # Nearest distances baseline vs optimized
    nearest_baseline = compute_nearest_distance_km(dlat, dlon, baseline_lat, baseline_lon)
    opt_lat = cand_lat[np.array(selected_idx, dtype=int)]
    opt_lon = cand_lon[np.array(selected_idx, dtype=int)]
    nearest_opt = compute_nearest_distance_km(dlat, dlon, opt_lat, opt_lon)

    def pct(x: np.ndarray, q: float) -> float:
        x2 = x[np.isfinite(x)]
        return float(np.percentile(x2, q)) if len(x2) else float("nan")

    metrics = {
        "n_active_baseline": n_ativas,
        "p_selected": int(p_total),
        "n_candidates": int(len(cand)),
        "n_demand": int(nD),
        "radius_km": float(args.radius_km),
        "grid_deg": float(args.grid_deg),
        "min_sep_km": float(args.min_sep_km),
        "weight_mode": args.weight_mode,
        "coverage_baseline": coverage_baseline,
        "coverage_optimized": coverage_opt,
        "covered_weight_baseline": covered_weight_baseline,
        "covered_weight_optimized": covered_weight_opt,
        "gain_weight": covered_weight_opt - covered_weight_baseline,
        "nearest_baseline_km_mean": float(np.nanmean(nearest_baseline)),
        "nearest_optimized_km_mean": float(np.nanmean(nearest_opt)),
        "nearest_baseline_km_p50": pct(nearest_baseline, 50),
        "nearest_optimized_km_p50": pct(nearest_opt, 50),
        "nearest_baseline_km_p90": pct(nearest_baseline, 90),
        "nearest_optimized_km_p90": pct(nearest_opt, 90),
    }

    # ----------------------------
    # Save outputs
    # ----------------------------
    baseline_out = _fix_latlon_and_dedupe_cols(st.copy())
    baseline_out["site_type"] = "baseline_station"

    selected_sites = cand.iloc[selected_idx].copy()
    selected_sites["site_type"] = "selected_relocated"
    selected_out = _fix_latlon_and_dedupe_cols(selected_sites.copy())

    out_baseline_csv = os.path.join(args.out_dir, "baseline_stations.csv")
    out_selected_csv = os.path.join(args.out_dir, "selected_sites.csv")
    out_json = os.path.join(args.out_dir, "metrics.json")

    baseline_out.to_csv(out_baseline_csv, index=False)
    selected_out.to_csv(out_selected_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nWrote: {out_baseline_csv}\nWrote: {out_selected_csv}\nWrote: {out_json}")

    # ----------------------------
    # Plot map (selected only): PNG + PDF
    # ----------------------------
    out_png = os.path.join(args.out_dir, "map_selected_only.png")
    out_pdf = os.path.join(args.out_dir, "map_selected_only.pdf")
    try:
        plot_selected_only_map(
            selected_out_csv=out_selected_csv,
            cache_dir=args.map_cache_dir,
            out_png=out_png,
            out_pdf=out_pdf,
            png_dpi=int(args.map_png_dpi),
        )
        print(f"[INFO] Mapa salvo em: {out_png}")
        print(f"[INFO] Mapa salvo em: {out_pdf}")
    except Exception as e:
        print(f"[WARN] Não foi possível gerar o mapa automaticamente: {e}")
        print("[WARN] Para plotar o mapa, instale geopandas e matplotlib (e dependências do geopandas).")


if __name__ == "__main__":
    main()