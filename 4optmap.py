#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nome: Heder Dorneles Soares
Instituição: INPE
Laboratório: LIAREA (Laboratório de Inteligência Artificial para Aplicações AeroEspaciais e Ambientais)

Purpose
-------
Greedy solver for a Maximum Coverage Location Problem (MCLP)-style model consistent with the paper draft:

- Demand points D: regular grid over Brazil (optionally clipped to Brazil polygon from Natural Earth).
- Fixed stations E: current INCT active stations (always selected).
- Candidate sites C: Federal Institutes (IFs) from ifs-brasil.csv plus (optionally) existing stations as candidates,
  but fixed stations are already enforced.
- Coverage: a demand point i is covered if it is within radius r (km) of at least one selected station.
- Objective (MCLP): with a budget of k new sensors (or total p), maximize weighted covered demand.

Inputs (CSV)
------------
- inct_active_stations.csv: columns like [name, city, state, lat, lon]
- ifs-brasil.csv: columns include [Escola, UF, Município, Latitude, Longitude, ...]

Outputs
-------
- selected_sites.csv: fixed + selected new sites
- metrics.json: coverage metrics before/after
- optional map.png

Dependencies
------------
Required: numpy, pandas
Optional (for Brazil clipping / nicer map): geopandas, shapely, matplotlib
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Utilities
# ----------------------------

EARTH_RADIUS_KM = 6371.0088


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
    """
    Pairwise haversine distance (km) between (lat1,lon1) points and (lat2,lon2) points.
    Returns matrix shape (len(lat1), len(lat2)).
    """
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
    """
    Fix for pandas InvalidIndexError during concat:
    - Remove duplicated column names
    - Normalize lat/lon into Latitude/Longitude without creating duplicates
    """
    # ensure unique columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Normalize Latitude
    if "Latitude" in df.columns and "lat" in df.columns:
        df["Latitude"] = df["Latitude"].fillna(df["lat"])
        df.drop(columns=["lat"], inplace=True)
    elif "Latitude" not in df.columns and "lat" in df.columns:
        df.rename(columns={"lat": "Latitude"}, inplace=True)

    # Normalize Longitude
    if "Longitude" in df.columns and "lon" in df.columns:
        df["Longitude"] = df["Longitude"].fillna(df["lon"])
        df.drop(columns=["lon"], inplace=True)
    elif "Longitude" not in df.columns and "lon" in df.columns:
        df.rename(columns={"lon": "Longitude"}, inplace=True)

    # final pass: ensure string labels and unique columns
    df.columns = [str(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    return df


# ----------------------------
# Optional: Brazil polygon clipping
# ----------------------------

def try_load_brazil_polygon(cache_dir: str) -> Optional["object"]:
    """
    Attempts to download and load Natural Earth admin_0 countries and return Brazil geometry.
    Returns shapely geometry or None if geopandas/shapely unavailable or download fails.
    """
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
    """
    Build a regular lat/lon grid within bounds; optionally clip to Brazil polygon.
    bounds: (min_lon, min_lat, max_lon, max_lat)
    """
    min_lon, min_lat, max_lon, max_lat = bounds

    lats = np.arange(min_lat, max_lat + 1e-9, grid_deg)
    lons = np.arange(min_lon, max_lon + 1e-9, grid_deg)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    pts = pd.DataFrame(
        {"lat": lat_grid.ravel().astype(float), "lon": lon_grid.ravel().astype(float)}
    )

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
# Model / Solver
# ----------------------------

@dataclass
class Solution:
    selected_candidate_idx: list
    coverage_before: float
    coverage_after: float
    covered_weight_before: float
    covered_weight_after: float
    n_demand: int


def greedy_mclp(
    cover_cand: np.ndarray,      # shape (nD, nC), bool
    weights: np.ndarray,         # shape (nD,), float
    covered_by_fixed: np.ndarray,# shape (nD,), bool
    k_add: int,
    min_sep_km: float,
    cand_lat: np.ndarray,
    cand_lon: np.ndarray,
    fixed_lat: np.ndarray,
    fixed_lon: np.ndarray,
) -> list:
    """
    Greedy maximum-coverage selection:
    iteratively pick candidate with maximum additional covered weight.
    Enforces minimum separation between:
      - new selections and fixed stations
      - new selections among themselves
    """
    nD, nC = cover_cand.shape
    selected: list[int] = []
    covered = covered_by_fixed.copy()

    cover_uint8 = cover_cand.astype(np.uint8)

    invalid = np.zeros(nC, dtype=bool)
    if min_sep_km > 0 and len(fixed_lat) > 0:
        d_cf = haversine_matrix_km(cand_lat, cand_lon, fixed_lat, fixed_lon)  # (nC, nF)
        invalid |= (d_cf.min(axis=1) < min_sep_km)

    cand_cand_dist = None
    min_dist_to_selected = np.full(nC, np.inf, dtype=float)
    if min_sep_km > 0:
        cand_cand_dist = haversine_matrix_km(cand_lat, cand_lon, cand_lat, cand_lon)

    for _ in range(max(0, k_add)):
        uncovered_w = weights.copy()
        uncovered_w[covered] = 0.0

        gains = (cover_uint8.T @ uncovered_w).astype(float)

        gains[invalid] = -1.0
        if selected:
            gains[np.array(selected, dtype=int)] = -1.0

        if min_sep_km > 0 and cand_cand_dist is not None:
            too_close = min_dist_to_selected < min_sep_km
            gains[too_close] = -1.0

        j = int(np.argmax(gains))
        if gains[j] <= 0:
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
    """Return boolean array indicating whether each demand point is covered by any site within radius_km."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Greedy MCLP sensor placement (fixed + k new).")
    ap.add_argument("--stations_csv", default="inct_active_stations.csv", help="Existing stations CSV.")
    ap.add_argument("--ifs_csv", default="ifs-brasil.csv", help="Federal Institutes CSV.")
    ap.add_argument("--out_dir", default="out", help="Output directory.")
    ap.add_argument("--radius_km", type=float, default=250.0, help="Coverage radius r (km).")
    ap.add_argument("--grid_deg", type=float, default=0.5, help="Demand grid resolution in degrees.")
    ap.add_argument("--k_add", type=int, default=20, help="Number of new sensors to add.")
    ap.add_argument("--p_total", type=int, default=None, help="Total sensors (fixed + new). Overrides k_add if set.")
    ap.add_argument("--min_sep_km", type=float, default=50.0, help="Minimum separation (km) between installations.")
    ap.add_argument(
        "--weight_mode",
        choices=["uniform", "inverse_distance_to_fixed"],
        default="uniform",
        help="Demand weighting scheme.",
    )
    ap.add_argument("--weight_alpha", type=float, default=1.0, help="Alpha for inverse-distance weighting.")
    ap.add_argument("--clip_brazil", action="store_true", help="Clip demand grid to Brazil polygon (Natural Earth download).")
    ap.add_argument("--cache_dir", default=".cache_geo", help="Cache directory for Natural Earth downloads.")
    ap.add_argument("--plot", action="store_true", help="Save a simple lon/lat scatter plot.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(args.cache_dir)

    # ----------------------------
    # Load CSVs
    # ----------------------------
    st = read_csv_robust(args.stations_csv).copy()
    ifs = read_csv_robust(args.ifs_csv).copy()

    # Normalize station columns
    if "lat" in st.columns and "lon" in st.columns:
        st["lat"] = to_float_series(st["lat"])
        st["lon"] = to_float_series(st["lon"])
    else:
        for latc in ["Latitude", "LAT", "Lat", "latitude"]:
            for lonc in ["Longitude", "LON", "Lon", "longitude"]:
                if latc in st.columns and lonc in st.columns:
                    st["lat"] = to_float_series(st[latc])
                    st["lon"] = to_float_series(st[lonc])
                    break

    st = st.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # Normalize IF columns
    if "Latitude" in ifs.columns and "Longitude" in ifs.columns:
        ifs["lat"] = to_float_series(ifs["Latitude"])
        ifs["lon"] = to_float_series(ifs["Longitude"])
    else:
        raise ValueError("ifs-brasil.csv must contain Latitude and Longitude columns.")

    ifs = ifs.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # Candidate set C: ONLY Federal Institutes (IFs)
    cand = ifs.copy()
    cand["site_type"] = "IF_candidate"

    # Fixed stations E
    fixed = st.copy()
    fixed["site_type"] = "fixed_station"

    fixed_lat = fixed["lat"].to_numpy(dtype=float)
    fixed_lon = fixed["lon"].to_numpy(dtype=float)
    cand_lat = cand["lat"].to_numpy(dtype=float)
    cand_lon = cand["lon"].to_numpy(dtype=float)

    # Budget
    if args.p_total is not None:
        k_add = max(0, int(args.p_total) - len(fixed))
    else:
        k_add = max(0, int(args.k_add))

    # ----------------------------
    # Demand points D
    # ----------------------------
    all_lat = np.concatenate([fixed_lat, cand_lat]) if len(fixed_lat) else cand_lat
    all_lon = np.concatenate([fixed_lon, cand_lon]) if len(fixed_lon) else cand_lon

    if len(all_lat) == 0 or len(all_lon) == 0:
        bounds = (-74.0, -34.0, -34.0, 6.0)
    else:
        min_lat = float(np.nanmin(all_lat)) - 1.0
        max_lat = float(np.nanmax(all_lat)) + 1.0
        min_lon = float(np.nanmin(all_lon)) - 1.0
        max_lon = float(np.nanmax(all_lon)) + 1.0
        bounds = (max(-74.5, min_lon), max(-35.0, min_lat), min(-33.5, max_lon), min(6.5, max_lat))

    brazil_poly = None
    if args.clip_brazil:
        brazil_poly = try_load_brazil_polygon(args.cache_dir)

    demand = build_demand_grid(bounds=bounds, grid_deg=args.grid_deg, brazil_poly=brazil_poly)
    dlat = demand["lat"].to_numpy(dtype=float)
    dlon = demand["lon"].to_numpy(dtype=float)
    nD = len(demand)

    # ----------------------------
    # Weights
    # ----------------------------
    weights = np.ones(nD, dtype=float)

    covered_fixed = compute_covered_by_sites(dlat, dlon, fixed_lat, fixed_lon, args.radius_km)

    if args.weight_mode == "inverse_distance_to_fixed":
        nearest_fixed = compute_nearest_distance_km(dlat, dlon, fixed_lat, fixed_lon)
        med = float(np.nanmedian(nearest_fixed[np.isfinite(nearest_fixed)])) if np.isfinite(nearest_fixed).any() else 1.0
        weights = 1.0 + args.weight_alpha * (nearest_fixed / max(med, 1e-6))

    total_weight = float(weights.sum())
    covered_weight_before = float(weights[covered_fixed].sum())
    coverage_before = covered_weight_before / total_weight if total_weight > 0 else 0.0

    # ----------------------------
    # Coverage matrix a_ij for candidates
    # ----------------------------
    dist_dc = haversine_matrix_km(dlat, dlon, cand_lat, cand_lon)
    cover_cand = dist_dc <= float(args.radius_km)

    # ----------------------------
    # Solve via greedy MCLP
    # ----------------------------
    selected_idx = greedy_mclp(
        cover_cand=cover_cand,
        weights=weights,
        covered_by_fixed=covered_fixed,
        k_add=k_add,
        min_sep_km=float(args.min_sep_km),
        cand_lat=cand_lat,
        cand_lon=cand_lon,
        fixed_lat=fixed_lat,
        fixed_lon=fixed_lon,
    )

    if selected_idx:
        covered_after = covered_fixed | cover_cand[:, selected_idx].any(axis=1)
    else:
        covered_after = covered_fixed.copy()

    covered_weight_after = float(weights[covered_after].sum())
    coverage_after = covered_weight_after / total_weight if total_weight > 0 else 0.0

    nearest_before = compute_nearest_distance_km(dlat, dlon, fixed_lat, fixed_lon)
    if selected_idx:
        new_lat = cand_lat[np.array(selected_idx, dtype=int)]
        new_lon = cand_lon[np.array(selected_idx, dtype=int)]
        all_lat2 = np.concatenate([fixed_lat, new_lat])
        all_lon2 = np.concatenate([fixed_lon, new_lon])
    else:
        all_lat2 = fixed_lat
        all_lon2 = fixed_lon

    nearest_after = compute_nearest_distance_km(dlat, dlon, all_lat2, all_lon2)

    def pct(x: np.ndarray, q: float) -> float:
        x2 = x[np.isfinite(x)]
        return float(np.percentile(x2, q)) if len(x2) else float("nan")

    metrics = {
        "n_fixed": int(len(fixed)),
        "n_candidates": int(len(cand)),
        "n_demand": int(nD),
        "radius_km": float(args.radius_km),
        "grid_deg": float(args.grid_deg),
        "k_add": int(k_add),
        "min_sep_km": float(args.min_sep_km),
        "weight_mode": args.weight_mode,
        "coverage_before": coverage_before,
        "coverage_after": coverage_after,
        "covered_weight_before": covered_weight_before,
        "covered_weight_after": covered_weight_after,
        "gain_weight": covered_weight_after - covered_weight_before,
        "nearest_before_km_mean": float(np.nanmean(nearest_before)),
        "nearest_after_km_mean": float(np.nanmean(nearest_after)),
        "nearest_before_km_p50": pct(nearest_before, 50),
        "nearest_after_km_p50": pct(nearest_after, 50),
        "nearest_before_km_p90": pct(nearest_before, 90),
        "nearest_after_km_p90": pct(nearest_after, 90),
    }

    # ----------------------------
    # Save outputs
    # ----------------------------
    new_sites = cand.iloc[selected_idx].copy() if selected_idx else cand.iloc[[]].copy()
    new_sites["site_type"] = "selected_new"

    # ===== MERGED FIX HERE =====
    fixed_out = _fix_latlon_and_dedupe_cols(fixed.copy())
    new_out = _fix_latlon_and_dedupe_cols(new_sites.copy())

    combined = pd.concat([fixed_out, new_out], ignore_index=True, sort=False)
    combined = combined.loc[:, ~combined.columns.duplicated()].copy()
    # ===========================

    out_csv = os.path.join(args.out_dir, "selected_sites.csv")
    combined.to_csv(out_csv, index=False)

    out_json = os.path.join(args.out_dir, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if args.plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            plt.figure()
            plt.scatter(dlon, dlat, s=5)
            plt.scatter(fixed_lon, fixed_lat, marker="^", s=80)
            if selected_idx:
                plt.scatter(new_lon, new_lat, marker="*", s=120)
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title("Demand grid, fixed stations, and selected new IF sites")
            plt.tight_layout()
            out_png = os.path.join(args.out_dir, "map.png")
            plt.savefig(out_png, dpi=200)
            plt.close()
        except Exception:
            pass

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nWrote: {out_csv}\nWrote: {out_json}")


if __name__ == "__main__":
    main()