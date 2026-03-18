import os
import zipfile
import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

CSV_FILE = "ifs-brasil.csv"
CACHE_DIR = "./ne_admin1_cache"
OUT_PNG = "mapa_ifs.png"

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

# =========================
# 1) Ler CSV
# =========================
df = read_csv_robust(CSV_FILE)

lat_col = pick_col(df, ["LATITUDE", "latitude", "lat", "y"])
lon_col = pick_col(df, ["LONGITUDE", "longitude", "lon", "lng", "x"])

if not lat_col or not lon_col:
    raise SystemExit(
        "Não encontrei colunas de latitude/longitude.\n"
        f"Colunas encontradas: {list(df.columns)}\n"
        "Diga quais são as colunas de lat/lon que eu ajusto."
    )

df[lat_col] = to_float_series(df[lat_col])
df[lon_col] = to_float_series(df[lon_col])

df = df.dropna(subset=[lat_col, lon_col]).copy()
df = df[~((df[lat_col] == 0) & (df[lon_col] == 0))].copy()
df = df.drop_duplicates(subset=[lat_col, lon_col]).copy()

print(f"Arquivo: {CSV_FILE}")
print(f"Colunas usadas: lat={lat_col} lon={lon_col}")
print(f"Pontos após limpeza/dedup: {len(df)}")

gdf_pts = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
    crs="EPSG:4326"
)

# =========================
# 2) Baixar limites estaduais (Natural Earth admin_1) em cache local
# =========================
os.makedirs(CACHE_DIR, exist_ok=True)
zip_path = os.path.join(CACHE_DIR, "ne_admin1.zip")
shp_path = os.path.join(CACHE_DIR, "ne_10m_admin_1_states_provinces.shp")

url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip"

if not os.path.exists(zip_path):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(r.content)

if not os.path.exists(shp_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(CACHE_DIR)

states = gpd.read_file(shp_path).to_crs("EPSG:4326")

# =========================
# 3) Filtrar Brasil (robusto)
# =========================
print("Colunas do shapefile (amostra):", list(states.columns)[:15])

br_states = None

# tenta 'admin' == 'Brazil'
if "admin" in states.columns:
    br_states = states[states["admin"] == "Brazil"].copy()

# fallback: ISO BRA (adm0_a3)
if (br_states is None or len(br_states) == 0) and "adm0_a3" in states.columns:
    br_states = states[states["adm0_a3"] == "BRA"].copy()

# fallback: ISO_A2 BR (se existir)
if (br_states is None or len(br_states) == 0) and "iso_a2" in states.columns:
    br_states = states[states["iso_a2"] == "BR"].copy()

if br_states is None or len(br_states) == 0:
    raise SystemExit(
        "Não consegui filtrar os estados do Brasil no shapefile.\n"
        "Me mande a lista completa de colunas do shapefile (print(states.columns)) que eu ajusto o filtro."
    )

print(f"Estados do Brasil encontrados no shapefile: {len(br_states)}")

# =========================
# 4) Plot e salvar PNG
# =========================
fig, ax = plt.subplots(figsize=(10, 10))

# desenha fronteiras estaduais (sem preenchimento)
br_states.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.6)

# desenha pontos
gdf_pts.plot(ax=ax, markersize=10, alpha=0.7)

# ajusta limites para enquadrar Brasil (usando bounds do shapefile)
minx, miny, maxx, maxy = br_states.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

ax.set_title("Institutos Federais do Brasil — campi/unidades do CSV")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
print(f"Mapa salvo em: {OUT_PNG}")

# Se você estiver em ambiente com GUI, pode descomentar:
plt.show()