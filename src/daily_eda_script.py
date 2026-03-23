import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import argparse
import pyarrow.fs as fs

import h3
import geopandas as gpd

import pandas as pd
import boto3

from eda_utils import compute_user_daily_rollup, compute_hex_daily_rollup
from datetime import timedelta



def has_dask_parquet_dataset_fast_s3(path, s3):

    path = path.replace("s3://", "").rstrip("/")

    try:
        top = s3.get_file_info(fs.FileSelector(path, recursive=False))

        for info in top:
            if info.is_file and info.path.endswith(".parquet"):
                return True
            if info.path.endswith("/_metadata") or info.path.endswith("/_common_metadata"):
                return True

        # Check one partition level
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

    parser = argparse.ArgumentParser(description="Process some arguments.")

    # Mandatory arguments
    parser.add_argument("--date", type=str, required=True, help="Date to analyze, e.g. 2026-01-21")
    parser.add_argument("--country", type=str, required=True, help="Country code of the country to process")
    parser.add_argument("--save_folder", type=str, required=True,
                    help="Folder where results will be saved")
    parser.add_argument("--tmz", type=str, required=True, help="Timezone to apply (e.g., 'Europe/Istanbul')")
    
    # Optional arguments
    parser.add_argument( "--d_before", type=int, default=-1, help="Number of days before the selected date (default: -1)")
    parser.add_argument( "--d_after", type=int, default=7, help="Number of days after the selected date (default: 7)")

    parser.add_argument( "--h3res", type=int, default=7, help="Default H3 resolution")
        
    parser.add_argument("--spatial_filter", type=str, default="", help="Path to a shapefile for spatial filtering")

    args = parser.parse_args()

    session = boto3.Session(profile_name = 'ECA')
    s3 = fs.S3FileSystem(
                region = session.region_name, 
                access_key = session.get_credentials().access_key,
                secret_key = session.get_credentials().secret_key, 
                session_token = session.get_credentials().token,
    )


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


    # If I want to process day d: load d-1 (time-zone) and x+1, ..., x+7
    dates_to_load = [dt + timedelta(days=i) for i in range(args.d_before, args.d_after+1)]
    

    paths_to_load = [path_base+f"/year={d.year}/date={d.strftime('%Y-%m-%d')}/*.parquet" for d in dates_to_load]
    paths_to_load = [p for p in paths_to_load if has_dask_parquet_dataset_fast_s3(p.replace("*.parquet",""), s3)]

    #print("Day to analyze", dt.strftime('%Y-%m-%d'))
    #print(f"window {args.d_before}, {args.d_after}\n")
    if len(paths_to_load)>0:
        print("Datasets to load:")
        print(*paths_to_load, sep="\n")
    else:
        print("No data for the selected date and window")
        return

    # load the shapefile
    apply_spatial_filtering = False
    if shapefile_path!="":
        shape_filtering = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
        apply_spatial_filtering = True
        shape_filtering = shape_filtering[["geometry"]]
        minx, miny, maxx, maxy = shape_filtering.total_bounds


    # 0. Load the raw-gps dataset
    columns = ['uid', 'datetime', 'hex_id', 'latitude', 'longitude', 'country']
    ddf = dd.read_parquet(paths_to_load, columns=columns, storage_options={"profile": "ECA"})

    # 1. Bounding box filter
    if apply_spatial_filtering:
        ddf = ddf[(ddf.longitude >= minx) & (ddf.longitude <= maxx) &
            (ddf.latitude  >= miny) & (ddf.latitude  <= maxy)]

    # 2. Process datetime (lazy)
    ddf['local_datetime'] = (
    dd.to_datetime(ddf['datetime'], utc=True)
      .dt.tz_convert(timezone_conversion)
      .dt.tz_localize(None))

    # Filter by day: useful as each date may contain observation from previous days
    start = pd.to_datetime(DATE)
    #end = start + timedelta(days=1)

    ddf["date"] = ddf["local_datetime"].dt.date
    ddf = ddf[ddf["date"] == start.date()]

    # Pandas is way faster for 3-15 M rows (avg, obs/day). So this part will be done in Pandas.

    # materialize the dataset
    with ProgressBar():
        print("Computing Dask dataframe...")
        df = ddf.compute()
        print("Rows loaded:", len(df))

    if len(df)==0:
        print("No data in df")
        return

    # execute the exact spatial filtering
    if apply_spatial_filtering:
        print(f"Rows before spatial join: {len(df)}")
        gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
        df = gpd.sjoin(gdf_points, shape_filtering, predicate="within", how="inner")
        print(f"Rows after spatial join : {len(df)}")



    # set the desired H3 resolution
    list_h3_cell_ids = [h3.latlng_to_cell(lat, lon, H3_res) for lat, lon in zip(df["latitude"].values, df["longitude"].values)]
    df["hex_id"] = list_h3_cell_ids
            

    # 1. Temporal - Pandas
    df["hour"] = df[dat_col].dt.hour
    DF_temporal_stats = df.groupby(["date", "hour"]).agg(n_points=("uid", "size"), n_users=("uid", "nunique"), n_hexes=("hex_id", "nunique")).reset_index()

    # 2. Users - Pandas
    DF_users_stats = df.groupby(["date", "uid"]).agg(
                            n_points=("hex_id", "size"),
                            n_hexes=("hex_id", "nunique"),
                            first_ts=(dat_col, "min"),
                            last_ts=(dat_col, "max")
                            ).reset_index()

    DF_users_stats["active_window_s"] = (DF_users_stats["last_ts"] - DF_users_stats["first_ts"]).dt.total_seconds()
    DF_users_stats["active_window_m"] = DF_users_stats["active_window_s"]/60
    DF_users_stats = DF_users_stats[["date", "uid", "n_points", "n_hexes", "active_window_m"]]

    # 3. Spatial - Pandas
    DF_spatial_stats = df.groupby(["date", "hex_id"]).agg(
        n_points=("uid", "size"),
        n_users=("uid", "nunique")
    ).reset_index()

    # 4 Daily Roll-UP USER
    DF_roll_up_user = compute_user_daily_rollup(DF_users_stats)
    
    # 5 Daily Roll-UP SPATIAL
    DF_roll_up_spatial = compute_hex_daily_rollup(DF_spatial_stats)


    # Save the computed dataframes (.parquet)

    # 1) temporal dataset
    path_save = f'{base_path_save}temporal_stats/'
    ddf_temporal = dd.from_pandas(DF_temporal_stats, npartitions=1)  # 1 partition is fine for 24 rows
    ddf_temporal.to_parquet(path_save, partition_on=["date"], write_index=False, engine="pyarrow", storage_options= {'profile':'ECA'})
    print(f"Daily temporal dataset saved ({len(DF_temporal_stats)} rows) -> {path_save}")

    # 2) user dataset
    path_save = f'{base_path_save}user_stats/'
    ddf_user = dd.from_pandas(DF_users_stats, npartitions=1)
    ddf_user.to_parquet(path_save, partition_on=["date"], write_index=False, engine="pyarrow", storage_options= {'profile':'ECA'})
    print(f"Daily user dataset saved ({len(DF_users_stats)} rows) -> {path_save}")

    # 3) spatial dataset
    path_save = f'{base_path_save}spatial_stats/'
    ddf_spatial = dd.from_pandas(DF_spatial_stats, npartitions=1)
    ddf_spatial.to_parquet(path_save, partition_on=["date"], write_index=False, engine="pyarrow", storage_options= {'profile':'ECA'})
    print(f"Daily spatial dataset saved ({len(DF_spatial_stats)} rows) -> {path_save}")

    # 4) roll-up user
    path_save = f'{base_path_save}user_daily_summary/'
    ddf_roll_up_user = dd.from_pandas(DF_roll_up_user, npartitions=1)
    ddf_roll_up_user.to_parquet(path_save, partition_on=["date"], write_index=False, engine="pyarrow", storage_options= {'profile':'ECA'})
    print(f"Daily summary user ({len(DF_roll_up_user)} rows) -> {path_save}")

    # 5) roll-up spatial
    path_save = f'{base_path_save}spatial_daily_summary/'
    ddf_roll_up_spatial = dd.from_pandas(DF_roll_up_spatial, npartitions=1)
    ddf_roll_up_spatial.to_parquet(path_save, partition_on=["date"], write_index=False, engine="pyarrow", storage_options= {'profile':'ECA'})
    print(f"Daily summary spatial ({len(DF_roll_up_spatial)} rows) -> {path_save}")



if __name__ == "__main__":
    main()

