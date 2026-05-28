"""
compute_daily_stats.py
-------------------
Daily GPS data processing pipeline for the ECA Resilience project.
 
For a given target date and country, this script:
  1. Loads raw Veraset GPS parquet data from S3 over a configurable date window
     (to handle UTC→local timezone shifts and Veraset feed delivery lags).
  2. Applies optional spatial filtering via a shapefile.
  3. Re-aggregates observations to the desired H3 resolution.
  4. Computes five daily statistics datasets and saves them to S3 as parquet.
 
Output datasets (partitioned by date):
  - temporal_stats        : hourly counts of points, users, and hexes
  - user_stats            : per-user daily points, hexes, active window
  - spatial_stats         : per-hex daily points and users
  - user_daily_summary    : 1-row daily rollup — user-level metrics
  - spatial_daily_summary : 1-row daily rollup — spatial concentration metrics
 
Usage:
    python compute_daily_stats.py \\
        --date 2023-04-20 \\
        --country PH \\
        --save_folder manila_apr2023 \\
        --tmz Asia/Manila
 
See README.md for the full argument reference and pipeline architecture.
"""

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import argparse
import pyarrow.fs as fs

import h3
import geopandas as gpd

import pandas as pd
import boto3

from rollup_utils import compute_user_daily_rollup, compute_hex_daily_rollup
from datetime import timedelta



def has_dask_parquet_dataset_fast_s3(path, s3):
    """
    Return True if a Dask-readable parquet dataset exists at the given S3 path.
 
    Uses a shallow (non-recursive) listing to avoid expensive full scans.
    Handles flat layouts, Dask metadata files, and one-level Hive partitions.
 
    Parameters
    ----------
    path : str
        S3 path to check (with or without s3:// prefix).
    s3 : fs.S3FileSystem
        Authenticated PyArrow S3 filesystem instance.
    """

    path = path.replace("s3://", "").rstrip("/")

    try:
        top = s3.get_file_info(fs.FileSelector(path, recursive=False))

        for info in top:
            if info.is_file and info.path.endswith(".parquet"):
                return True
            if info.path.endswith("/_metadata") or info.path.endswith("/_common_metadata"):
                return True

        # Check one partition level (Hive-partitioned layout)
        for info in top:
            if info.is_dir:
                sub = s3.get_file_info(fs.FileSelector(info.path, recursive=False))
                if any(x.is_file and x.path.endswith(".parquet") for x in sub):
                    return True
        return False
    except Exception as e:
        print(f"S3 check failed for {path}: {e}")
        return False


def main():

    parser = argparse.ArgumentParser(description="Compute daily EDA statistics from raw Veraset GPS data.")

    # Mandatory arguments
    parser.add_argument("--date", type=str, required=True, help="Target date to process (YYYY-MM-DD)")
    parser.add_argument("--country", type=str, required=True, help="ISO country code (e.g. PH, TR)")
    parser.add_argument("--save_folder", type=str, required=True,
                    help="Output folder name within the S3 base path")
    parser.add_argument("--tmz", type=str, required=True, help="Local timezone for UTC conversion (e.g. 'Asia/Manila')")
    
    # Optional arguments
    parser.add_argument( "--d_before", type=int, default=-1, help="Days before the target date to include in the load window (default: -1)")
    parser.add_argument( "--d_after", type=int, default=7, help="Days after the target date to include in the load window (default: 7)")

    parser.add_argument( "--h3res", type=int, default=7, help="H3 spatial resolution for aggregation (default: 7, ~5 km²)")
        
    parser.add_argument("--spatial_filter", type=str, default="", help="Path to a shapefile for spatial clipping (optional)")

    args = parser.parse_args()

    session = boto3.Session(profile_name = 'ECA')
    s3 = fs.S3FileSystem(
                region = session.region_name, 
                access_key = session.get_credentials().access_key,
                secret_key = session.get_credentials().secret_key, 
                session_token = session.get_credentials().token,
    )


    # --- Path configuration --
    base_path_save = f"s3://wbgggscecovid19dev-mobility/proposals/561/{args.save_folder}/"
    print("Result folder path:", base_path_save)

    # date-related info
    DATE = args.date
    dt = pd.to_datetime(DATE)
    dat_col = "local_datetime"
    timezone_conversion = args.tmz

    # spatial info
    country = args.country
    H3_res = args.h3res
    shapefile_path = args.spatial_filter

    # folder raw GPS dataset
    path_base = f"s3://wbgggscecovid19dev-mobility/veraset/country={country}"

    print("\n" + "="*65)
    print("RUN CONFIGURATION")
    print("="*65)
    
    print(f"Date to analyze        : {DATE}")
    print(f"Parsed datetime        : {dt}")
    print(f"Country                : {country}")
    print(f"Timezone               : {timezone_conversion}")
    
    print(f"Window (days)          : {args.d_before} -> {args.d_after}")
    print(f"H3 resolution          : {H3_res}")
    
    print(f"Raw dataset base path  : {path_base}")
    print(f"Output base path       : {base_path_save}")
    
    print(f"Spatial filter file    : {shapefile_path if shapefile_path else 'None'}")
    
    print("="*65 + "\n")


    # --- Build the date window to load ---
    # Veraset data is stored by UTC date. To correctly filter to a single local
    # calendar day, we must load:
    #   - d-1: observations near midnight that fall on the previous UTC date
    #          after timezone conversion
    #   - d+1 to d+7: Veraset feed delivery lag — observations from date d may
    #          appear in feeds delivered up to 3 days later; d+7 is a conservative
    #          margin that ensures full coverage.
    dates_to_load = [dt + timedelta(days=i) for i in range(args.d_before, args.d_after+1)]
    
    # Filter to paths that actually exist on S3 (avoids read errors on missing dates)
    paths_to_load = [path_base+f"/year={d.year}/date={d.strftime('%Y-%m-%d')}/*.parquet" for d in dates_to_load]
    paths_to_load = [p for p in paths_to_load if has_dask_parquet_dataset_fast_s3(p.replace("*.parquet",""), s3)]

    if len(paths_to_load)>0:
        print("Datasets to load:")
        print(*paths_to_load, sep="\n")
    else:
        print("No data for the selected date and window")
        return

    # --- Spatial filter setup ---
    apply_spatial_filtering = False
    if shapefile_path!="":
        shape_filtering = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
        apply_spatial_filtering = True
        shape_filtering = shape_filtering[["geometry"]]
        minx, miny, maxx, maxy = shape_filtering.total_bounds


    # --- Load raw GPS data via Dask (lazy) --
    columns = ['uid', 'datetime', 'hex_id', 'latitude', 'longitude', 'country']
    ddf = dd.read_parquet(paths_to_load, columns=columns, storage_options={"profile": "ECA"})

    # --- Bounding box pre-filter (fast, lazy) ---
    # Applied before the exact spatial join to reduce the data volume
    # that needs to be materialized. The exact polygon join follows later.
    if apply_spatial_filtering:
        ddf = ddf[(ddf.longitude >= minx) & (ddf.longitude <= maxx) &
            (ddf.latitude  >= miny) & (ddf.latitude  <= maxy)]

    # --- UTC → local datetime conversion (lazy) ---
    ddf['local_datetime'] = (
    dd.to_datetime(ddf['datetime'], utc=True)
      .dt.tz_convert(timezone_conversion)
      .dt.tz_localize(None))

    # --- Filter to the target local date (lazy) ---
    # Necessary because the load window spans multiple UTC dates.
    start = pd.to_datetime(DATE)
    #end = start + timedelta(days=1)

    ddf["date"] = ddf["local_datetime"].dt.date
    ddf = ddf[ddf["date"] == start.date()]

    # --- Materialize to Pandas ---
    # Pandas is significantly faster than Dask for the per-day volumes
    # typical in this dataset (3–15M rows/day after filtering).
    with ProgressBar():
        print("Computing Dask dataframe...")
        df = ddf.compute()
        print("Rows loaded:", len(df))

    if len(df)==0:
        print("No data in df")
        return

    # --- Exact spatial join (Pandas/GeoPandas) ---
    if apply_spatial_filtering:
        print(f"Rows before spatial join: {len(df)}")
        gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
        df = gpd.sjoin(gdf_points, shape_filtering, predicate="within", how="inner")
        print(f"Rows after spatial join : {len(df)}")


    # --- Re-aggregate to the desired H3 resolution ---
    # The raw data uses H3 resolution 7 by default. This step recomputes
    # hex IDs from lat/lon coordinates to support any target resolution.
    list_h3_cell_ids = [h3.latlng_to_cell(lat, lon, H3_res) for lat, lon in zip(df["latitude"].values, df["longitude"].values)]
    df["hex_id"] = list_h3_cell_ids

    # --- Compute daily statistics ---

    # 1. Temporal: hourly counts of points, users, and active hexes
    df["hour"] = df[dat_col].dt.hour
    DF_temporal_stats = df.groupby(["date", "hour"]).agg(n_points=("uid", "size"), n_users=("uid", "nunique"), n_hexes=("hex_id", "nunique")).reset_index()

    # 2. User-level: per-user daily activity (points, hexes, active window)
    DF_users_stats = df.groupby(["date", "uid"]).agg(
                            n_points=("hex_id", "size"),
                            n_hexes=("hex_id", "nunique"),
                            first_ts=(dat_col, "min"),
                            last_ts=(dat_col, "max")
                            ).reset_index()

    DF_users_stats["active_window_s"] = (DF_users_stats["last_ts"] - DF_users_stats["first_ts"]).dt.total_seconds()
    DF_users_stats["active_window_m"] = DF_users_stats["active_window_s"]/60
    DF_users_stats = DF_users_stats[["date", "uid", "n_points", "n_hexes", "active_window_m"]]

    # 3. Spatial: per-hex daily points and unique users
    DF_spatial_stats = df.groupby(["date", "hex_id"]).agg(
        n_points=("uid", "size"),
        n_users=("uid", "nunique")
    ).reset_index()

    # 4. Daily user rollup: 1-row summary with percentiles and left-tail count
    DF_roll_up_user = compute_user_daily_rollup(DF_users_stats)
    
    # 5. Daily spatial rollup: 1-row summary with concentration metrics
    DF_roll_up_spatial = compute_hex_daily_rollup(DF_spatial_stats)


    # --- Save all datasets to S3 as partitioned parquet ---

    def save(df: pd.DataFrame, subfolder: str, label: str) -> None:
        path = f"{base_path_save}{subfolder}/"
        dd.from_pandas(df, npartitions=1).to_parquet(
            path,
            partition_on=["date"],
            write_index=False,
            engine="pyarrow",
            storage_options={"profile": "ECA"}
        )
        print(f"{label} ({len(df)} rows) -> {path}")
 
    save(DF_temporal_stats,  "temporal_stats",        "Daily temporal dataset saved")
    save(DF_users_stats,     "user_stats",             "Daily user dataset saved")
    save(DF_spatial_stats,   "spatial_stats",          "Daily spatial dataset saved")
    save(DF_roll_up_user,    "user_daily_summary",     "Daily user rollup saved")
    save(DF_roll_up_spatial, "spatial_daily_summary",  "Daily spatial rollup saved")




if __name__ == "__main__":
    main()

