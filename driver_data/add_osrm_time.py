from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm                 
import requests, pandas as pd, time, math


PROJECT_ROOT = Path.cwd().resolve().parents[0]     
COMBINED_DIR   = PROJECT_ROOT / "combined_path"
LARGER_DIR = COMBINED_DIR / "new_test" / "larger"
CELL_FILE_ADDITION = "larger_0125"

PARQUET_IN = LARGER_DIR / f"trips_with_price_duration_{CELL_FILE_ADDITION}_km.parquet"
PARQUET_OUT = LARGER_DIR / f"trips_with_price_duration_{CELL_FILE_ADDITION}_km_osrm.parquet"

N_THREADS   = 40                   
OSRM_FMT    = "http://localhost:5000/route/v1/driving/{lon1:.6f},{lat1:.6f};{lon2:.6f},{lat2:.6f}?overview=false"
NEAREST_FMT = "http://localhost:5000/nearest/v1/driving/{lon:.6f},{lat:.6f}"

df = pd.read_parquet(PARQUET_IN).reset_index(drop=True)
print(f"Loaded {len(df):,} rows")


df["osrm_sec"] = pd.NA
df["osrm_km"]  = pd.NA


def osrm_call(row, retries=3):
    url = OSRM_FMT.format(lon1=row.begin_lng, lat1=row.begin_lat,
                          lon2=row.end_lng,   lat2=row.end_lat)
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            data = r.json()["routes"][0]
            return data["duration"], data["distance"] / 1000.0  
        except Exception:
            time.sleep(0.5)
    return math.nan, math.nan


with ThreadPoolExecutor(max_workers=N_THREADS) as pool, tqdm(total=len(df)) as bar:
    futures = {pool.submit(osrm_call, row): idx
               for idx, row in enumerate(df.itertuples(index=False))}

    for fut in as_completed(futures):
        idx = futures[fut]
        sec, km = fut.result()
        df.iat[idx, df.columns.get_loc("osrm_sec")] = sec
        df.iat[idx, df.columns.get_loc("osrm_km")]  = km
        bar.update()


df["osrm_sec"] = df["osrm_sec"].astype("float32")
df["osrm_km"]  = df["osrm_km"].astype("float32")

print(df[["duration_sec", "osrm_sec", "trip_distance_km", "osrm_km"]].describe())


def osrm_nearest(lat, lon, retries=3):
    """Return (lon_snap, lat_snap). If all retries fail, return (nan, nan)."""
    for _ in range(retries):
        try:
            r = requests.get(NEAREST_FMT.format(lon=lon, lat=lat), timeout=5)
            r.raise_for_status()
            wp = r.json()["waypoints"][0]["location"]  
            return float(wp[0]), float(wp[1])
        except Exception:
            time.sleep(0.5)
    return math.nan, math.nan

def osrm_route_from_coords(lon1, lat1, lon2, lat2, retries=2):
    for _ in range(retries):
        try:
            r = requests.get(
                OSRM_FMT.format(lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2),
                timeout=5
            )
            r.raise_for_status()
            data = r.json()["routes"][0]
            return data["duration"], data["distance"] / 1000.0 
        except Exception:
            time.sleep(0.5)
    return math.nan, math.nan


missing_mask = df["osrm_sec"].isna() | df["osrm_km"].isna()
todo_idx = df.index[missing_mask]
print(f"Retrying {len(todo_idx):,} rows via /nearest snap...")

def retry_row(idx):
    row = df.loc[idx]
    sec, km = osrm_call(row, retries=1)
    if not (math.isnan(sec) or math.isnan(km)):
        return idx, sec, km, math.nan, math.nan, math.nan, math.nan, False  

    lon1, lat1 = osrm_nearest(row.begin_lat, row.begin_lng)
    lon2, lat2 = osrm_nearest(row.end_lat,   row.end_lng)
    if any(math.isnan(v) for v in (lon1, lat1, lon2, lat2)):
        return idx, math.nan, math.nan, lon1, lat1, lon2, lat2, True

    sec2, km2 = osrm_route_from_coords(lon1, lat1, lon2, lat2, retries=2)
    return idx, sec2, km2, lon1, lat1, lon2, lat2, True

with ThreadPoolExecutor(max_workers=min(N_THREADS, 40)) as pool, tqdm(total=len(todo_idx)) as bar:
    futures = {pool.submit(retry_row, idx): idx for idx in todo_idx}
    for fut in as_completed(futures):
        idx, sec, km, sblon, sblat, selon, selat, snapped = fut.result()
        if not math.isnan(sec):
            df.iat[idx, df.columns.get_loc("osrm_sec")] = sec
            df.iat[idx, df.columns.get_loc("osrm_km")]  = km
        if snapped:
            df.iat[idx, df.columns.get_loc("snap_begin_lon")] = sblon
            df.iat[idx, df.columns.get_loc("snap_begin_lat")] = sblat
            df.iat[idx, df.columns.get_loc("snap_end_lon")]   = selon
            df.iat[idx, df.columns.get_loc("snap_end_lat")]   = selat
        bar.update()

remain = df["osrm_sec"].isna() | df["osrm_km"].isna()

df.drop(columns=["snap_begin_lon","snap_begin_lat","snap_end_lon","snap_end_lat"])
print(f"Still missing after retry: {remain.sum()} rows")



df.to_parquet(PARQUET_OUT, compression="zstd")
print("Saved →", PARQUET_OUT)

print(df[["duration_sec", "osrm_sec", "trip_distance_km", "osrm_km"]].describe())



