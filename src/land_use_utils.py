"""
land_use_utils.py
-----------------
Utility functions for spatial characterization of H3 hexagonal units.

Given a set of H3 hexagons (Area of Interest) and OpenStreetMap data
extracted from a Geofabrik PBF file, this module assigns to each hexagon:
  - a dominant land-use label (residential, commercial, industrial, etc.)
  - POI-based functional layers (counts of schools, hospitals, parks, etc.)
  - a binary highway presence indicator (motorway / trunk)

Typical usage (see LandUsage.ipynb):
    polys  = download_and_prepare_landuse_polys(osm, boundary_geom)
    pois   = download_and_prepare_pois(osm, boundary_geom)
    bldgs  = download_and_prepare_buildings(osm, boundary_geom)

    gdf_h3 = assign_land_use(gdf_h3, polys, bldgs)
    gdf_h3 = assign_poi_layers(gdf_h3, polys, pois)
    gdf_h3 = assign_highway_layer(osm, gdf_h3)
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# Helper: circular AOI
# ---------------------------------------------------------------------------

def circle_gdf(lat: float, lng: float, radius_km: float, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Return a single-row GeoDataFrame containing a circle of the given radius.

    The buffer is computed in a locally appropriate UTM projection to ensure
    metric accuracy, then reprojected to the requested CRS.

    Parameters
    ----------
    lat, lng   : float  — centre coordinates (WGS84 decimal degrees)
    radius_km  : float  — radius in kilometres
    crs        : str    — output CRS (default EPSG:4326)
    """
    center = gpd.GeoDataFrame(
        {"lat": [lat], "lng": [lng]},
        geometry=[Point(lng, lat)],
        crs="EPSG:4326"
    )
    center_utm = center.to_crs(center.estimate_utm_crs())
    circle = center_utm.buffer(radius_km * 1000)
    return gpd.GeoDataFrame(geometry=circle, crs=center_utm.crs).to_crs(crs)


# ---------------------------------------------------------------------------
# OSM data extraction
# ---------------------------------------------------------------------------

def download_and_prepare_landuse_polys(osm, boundary_geom=None) -> gpd.GeoDataFrame:
    """
    Extract and clean land-use, natural, and leisure polygons from a PBF file.

    Reads the `landuse`, `natural`, and `leisure` OSM tags, clips to the
    boundary if provided, and returns only polygon geometries with a minimal
    set of columns needed for land-use classification.

    Parameters
    ----------
    osm           : pyrosm.OSM  — OSM reader initialised on the PBF file
    boundary_geom : shapely geometry, optional — clip boundary

    Returns
    -------
    gpd.GeoDataFrame with columns: ['osmid', 'landuse', 'natural', 'leisure', 'geometry']
    """
    print("Reading polygon layers (landuse/natural/leisure) from PBF...")

    polys = osm.get_data_by_custom_criteria(
        custom_filter={"landuse": True, "natural": True, "leisure": True},
        filter_type="keep",
        keep_nodes=False,
        keep_ways=True,
        keep_relations=True,
    )

    if boundary_geom is not None:
        polys = polys[polys.geometry.intersects(boundary_geom)].copy()

    # Keep only valid polygons
    polys = polys[polys.geometry.notna()].copy()
    polys = polys[polys.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    # Retain minimal columns and deduplicate by OSM id
    keep_poly = [c for c in ["id", "landuse", "natural", "leisure", "geometry"] if c in polys.columns]
    polys = polys[keep_poly].reset_index(drop=True)
    polys = polys.drop_duplicates(subset=["id"]).reset_index(drop=True)
    polys = polys.rename(columns={"id": "osmid"})

    print(f"Polygons loaded: {len(polys):,}")
    print("Geometry types:", polys.geom_type.value_counts().to_dict())
    print("Non-null tag counts:")
    for c in ["landuse", "natural", "leisure"]:
        if c in polys.columns:
            print(f"  {c}: {polys[c].notna().sum():,}")

    return polys


def download_and_prepare_pois(osm, boundary_geom=None) -> gpd.GeoDataFrame:
    """
    Extract and clean Points of Interest from a PBF file.

    Reads amenity, shop, office, tourism, leisure, railway, and aeroway tags.
    Polygon POIs are converted to representative interior points so that all
    output geometries are Point type (required for the spatial join in
    assign_poi_layers).

    Parameters
    ----------
    osm           : pyrosm.OSM  — OSM reader initialised on the PBF file
    boundary_geom : shapely geometry, optional — clip boundary

    Returns
    -------
    gpd.GeoDataFrame with Point geometries and OSM tag columns.
    """
    print("Reading POIs from PBF...")

    pois_raw = osm.get_pois(
        custom_filter={
            "amenity": True,
            "shop": True,
            "office": True,
            "tourism": True,
            "leisure": True,
            "railway": ["station"],
            "aeroway": ["aerodrome", "terminal"],
        }
    )

    if boundary_geom is not None:
        pois_raw = pois_raw[pois_raw.geometry.intersects(boundary_geom)].copy()

    pois_raw = pois_raw[pois_raw.geometry.notna()].copy()

    # Keep only relevant tag columns
    keep_poi = [c for c in [
        "id", "amenity", "shop", "office", "tourism", "leisure",
        "railway", "aeroway", "geometry"
    ] if c in pois_raw.columns]
    pois_raw = pois_raw[keep_poi].copy()

    # Ensure all expected tag columns exist even if pyrosm did not return them
    for col in ["amenity", "shop", "office", "tourism", "leisure", "railway", "aeroway"]:
        if col not in pois_raw.columns:
            pois_raw[col] = np.nan

    # Separate points and polygons; convert polygon POIs to interior points
    pois_pt   = pois_raw[pois_raw.geom_type.isin(["Point", "MultiPoint"])].copy()
    pois_poly = pois_raw[pois_raw.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    if len(pois_poly) > 0:
        pois_poly["geometry"] = pois_poly.geometry.representative_point()

    pois = pd.concat([pois_pt, pois_poly], ignore_index=True)
    pois = pois.drop_duplicates(subset=["id"]).reset_index(drop=True)
    pois = pois.rename(columns={"id": "osmid"})

    # Keep only POIs with at least one usable tag
    core_cols = ["amenity", "shop", "office", "tourism", "leisure"]
    has_core = np.zeros(len(pois), dtype=bool)
    for c in core_cols:
        has_core |= pois[c].notna().to_numpy()

    is_train   = pois["railway"].eq("station")
    is_airport = pois["aeroway"].isin(["aerodrome", "terminal"])

    pois = pois.loc[has_core | is_train | is_airport].reset_index(drop=True)

    print(f"POIs loaded: {len(pois):,}")
    return pois


def download_and_prepare_buildings(osm, boundary_geom=None) -> gpd.GeoDataFrame:
    """
    Extract and clean typed building footprints from a PBF file.

    Only fetches building types relevant for land-use classification
    (commercial, industrial, institutional). Used as a fallback for hexagons
    that receive no label from the polygon-based classification.

    Parameters
    ----------
    osm           : pyrosm.OSM  — OSM reader initialised on the PBF file
    boundary_geom : shapely geometry, optional — clip boundary

    Returns
    -------
    gpd.GeoDataFrame with Polygon geometries and columns: ['osmid', 'building', 'geometry']
    """
    BUILDING_TYPES = [
        "commercial", "retail", "industrial", "warehouse",
        "hospital", "school", "university", "office",
        "train_station", "transportation",
    ]

    print("Reading filtered buildings from PBF...")

    bldg_raw = osm.get_buildings(custom_filter={"building": BUILDING_TYPES})

    if boundary_geom is not None:
        bldg_raw = bldg_raw[bldg_raw.geometry.intersects(boundary_geom)].copy()

    keep_cols = [c for c in ["id", "building", "geometry"] if c in bldg_raw.columns]
    bldgs = (
        bldg_raw
        .loc[
            bldg_raw.geometry.notna() &
            bldg_raw.geom_type.isin(["Polygon", "MultiPolygon"]),
            keep_cols
        ]
        .copy()
    )

    bldgs = bldgs.drop_duplicates(subset=["id"]).reset_index(drop=True)
    bldgs = bldgs.rename(columns={"id": "osmid"})

    print(f"Filtered buildings loaded: {len(bldgs):,}")
    print("Geometry types:", bldgs.geom_type.value_counts().to_dict())
    if "building" in bldgs.columns:
        print("\nTop building types:")
        print(bldgs["building"].value_counts().head(10))

    return bldgs


# ---------------------------------------------------------------------------
# Land-use classification helpers
# ---------------------------------------------------------------------------

def classify_polygon(row) -> str | None:
    """
    Map a single OSM polygon row to a land-use class label.

    Priority order: landuse tag → natural tag → leisure tag.
    Returns None for polygons that do not match any known category,
    which causes them to be excluded from the overlay.
    """
    lu  = row.get("landuse")
    nat = row.get("natural")
    lei = row.get("leisure")

    if pd.notna(lu):
        if lu == "residential":
            return "residential"
        if lu in {"commercial", "retail"}:
            return "commercial"
        if lu == "industrial":
            return "industrial"
        if lu == "construction":
            return "construction"
        if lu in {"farmland", "meadow", "orchard", "vineyard", "greenhouse_horticulture"}:
            return "farmland"
        if lu == "reservoir":
            return "water"

    if pd.notna(nat):
        if nat in {"water", "wetland"}:
            return "water"
        if nat in {"wood", "grassland", "scrub"}:
            return "green"

    if pd.notna(lei) and lei in {"park", "garden"}:
        return "green"

    return None


def classify_building(row) -> str | None:
    """
    Map a single OSM building row to a land-use class label.

    Used as a fallback for hexagons with no polygon-based label.
    Only industrial and commercial building types are classified;
    all others return None and are ignored.
    """
    b = row.get("building")
    if pd.isna(b):
        return None
    if b in {"industrial", "warehouse"}:
        return "industrial"
    if b in {"commercial", "retail", "office"}:
        return "commercial"
    return None


# ---------------------------------------------------------------------------
# Land-use assignment
# ---------------------------------------------------------------------------

def assign_land_use(
    gdf_h3: gpd.GeoDataFrame,
    polys: gpd.GeoDataFrame,
    bldgs: gpd.GeoDataFrame,
    MIN_SHARE: float = 0.20,
    MIN_SHARE_BLDG: float = 0.0001,
    native_crs: str = "EPSG:4326",
    return_shares: bool = False
) -> gpd.GeoDataFrame:
    """
    Assign a dominant land-use label to each H3 hexagon.

    Two-pass classification:
      1. Polygon-based: overlay each hexagon with OSM land-use polygons,
         compute the area share of each class, and assign the dominant class
         if its share exceeds MIN_SHARE.
      2. Building fallback: for hexagons that received no polygon label,
         repeat the overlay using typed building footprints with the lower
         threshold MIN_SHARE_BLDG.
      3. Hexagons with no label from either pass are assigned "other".

    Parameters
    ----------
    gdf_h3         : H3 hexagon GeoDataFrame with column 'h3_index'
    polys          : output of download_and_prepare_landuse_polys()
    bldgs          : output of download_and_prepare_buildings()
    MIN_SHARE      : minimum area share for a polygon label to be accepted
    MIN_SHARE_BLDG : minimum area share for a building fallback label
    native_crs     : output CRS (default EPSG:4326)
    return_shares  : if True, attach per-class area share dicts to each hex

    Returns
    -------
    gpd.GeoDataFrame with new columns: 'land_use' and optionally
    'lu_poly', 'lu_poly_share', 'lu_bldg', 'lu_bldg_share',
    'poly_shares', 'bldg_shares'
    """

    polys  = polys.copy()
    bldgs  = bldgs.copy()
    gdf_h3 = gdf_h3.copy()

    # Tag each polygon and building with its land-use class
    polys["lu_class"] = polys.apply(classify_polygon, axis=1)
    bldgs["lu_class"] = bldgs.apply(classify_building, axis=1)

    # Drop unclassified features before the overlay
    polys = polys[polys["lu_class"].notna()].copy()
    bldgs = bldgs[bldgs["lu_class"].notna()].copy()

    # Reproject everything to a metric CRS for area calculations
    crs = polys.estimate_utm_crs()
    gdf_h3 = gdf_h3.to_crs(crs)
    polys  = polys.to_crs(crs)
    bldgs  = bldgs.to_crs(crs)

    # Fix any invalid geometries before overlay
    gdf_h3["geometry"] = gdf_h3.geometry.buffer(0)
    polys["geometry"]  = polys.geometry.buffer(0)
    bldgs["geometry"]  = bldgs.geometry.buffer(0)

    hex_area = gdf_h3.set_index("h3_index").geometry.area

    # --- Pass 1: polygon-based labels ---
    h3_poly = gpd.overlay(
        gdf_h3[["h3_index", "geometry"]],
        polys[["lu_class", "geometry"]],
        how="intersection"
    )

    poly_shares = None

    if not h3_poly.empty:
        # Dissolve intersection fragments by (hex, class) before computing areas
        h3_poly = (
            h3_poly[["h3_index", "lu_class", "geometry"]]
            .dissolve(by=["h3_index", "lu_class"])
            .reset_index()
        )
        h3_poly["area"] = h3_poly.geometry.area

        poly_agg = (
            h3_poly.groupby(["h3_index", "lu_class"], as_index=False)["area"].sum()
        )
        poly_agg["hex_area"] = poly_agg["h3_index"].map(hex_area)
        poly_agg["share"]    = poly_agg["area"] / poly_agg["hex_area"]

        if return_shares:
            poly_shares = (
                poly_agg.groupby("h3_index")
                .apply(lambda df: dict(zip(df["lu_class"], df["share"])))
                .rename("poly_shares")
                .reset_index()
            )

        # Pick the dominant class per hexagon; discard if below MIN_SHARE
        idx = poly_agg.groupby("h3_index")["area"].idxmax()
        poly_labels = poly_agg.loc[idx]
        poly_labels = poly_labels[poly_labels["share"] >= MIN_SHARE]
        poly_labels = poly_labels.rename(columns={"lu_class": "lu_poly", "share": "lu_poly_share"})
        poly_labels = poly_labels[["h3_index", "lu_poly", "lu_poly_share"]]
    else:
        poly_labels = pd.DataFrame(columns=["h3_index", "lu_poly", "lu_poly_share"])

    # --- Pass 2: building fallback for unlabelled hexagons ---
    unlabeled = gdf_h3.loc[~gdf_h3["h3_index"].isin(poly_labels["h3_index"])]

    h3_bldg = gpd.overlay(
        unlabeled[["h3_index", "geometry"]],
        bldgs[["lu_class", "geometry"]],
        how="intersection"
    )

    bldg_shares = None

    if not h3_bldg.empty:
        h3_bldg = (
            h3_bldg[["h3_index", "lu_class", "geometry"]]
            .dissolve(by=["h3_index", "lu_class"])
            .reset_index()
        )
        h3_bldg["area"] = h3_bldg.geometry.area

        bldg_agg = (
            h3_bldg.groupby(["h3_index", "lu_class"], as_index=False)["area"].sum()
        )
        bldg_agg["hex_area"] = bldg_agg["h3_index"].map(hex_area)
        bldg_agg["share"]    = bldg_agg["area"] / bldg_agg["hex_area"]

        if return_shares:
            bldg_shares = (
                bldg_agg.groupby("h3_index")
                .apply(lambda df: dict(zip(df["lu_class"], df["share"])))
                .rename("bldg_shares")
                .reset_index()
            )

        idx = bldg_agg.groupby("h3_index")["area"].idxmax()
        bldg_labels = bldg_agg.loc[idx]
        bldg_labels = bldg_labels[bldg_labels["share"] >= MIN_SHARE_BLDG]
        bldg_labels = bldg_labels.rename(columns={"lu_class": "lu_bldg", "share": "lu_bldg_share"})
        bldg_labels = bldg_labels[["h3_index", "lu_bldg", "lu_bldg_share"]]
    else:
        bldg_labels = pd.DataFrame(columns=["h3_index", "lu_bldg", "lu_bldg_share"])

    # --- Merge labels and resolve final land_use ---
    gdf_h3 = gdf_h3.merge(poly_labels, on="h3_index", how="left")
    gdf_h3 = gdf_h3.merge(bldg_labels, on="h3_index", how="left")

    if return_shares:
        if poly_shares is not None:
            gdf_h3 = gdf_h3.merge(poly_shares, on="h3_index", how="left")
        if bldg_shares is not None:
            gdf_h3 = gdf_h3.merge(bldg_shares, on="h3_index", how="left")

    # Polygon label takes priority; building label as fallback; "other" if neither
    gdf_h3["land_use"] = (
        gdf_h3["lu_poly"]
        .fillna(gdf_h3["lu_bldg"])
        .fillna("other")
    )

    gdf_h3 = gdf_h3.to_crs(native_crs)
    gdf_h3["land_use"] = gdf_h3["land_use"].fillna("other")

    return gdf_h3


# ---------------------------------------------------------------------------
# POI layer assignment
# ---------------------------------------------------------------------------

def classify_poi(row) -> str | None:
    """
    Map a single OSM POI row to a functional category column name.

    Returns a string like 'n_schools' or 'n_hospitals' that corresponds
    to the count column added by assign_poi_layers(). Returns None for
    POIs that do not match any tracked category.
    """
    amenity = row.get("amenity")
    shop    = row.get("shop")
    office  = row.get("office")
    tourism = row.get("tourism")
    railway = row.get("railway")
    aeroway = row.get("aeroway")

    if amenity == "school":       return "n_schools"
    if amenity == "hospital":     return "n_hospitals"
    if amenity == "university":   return "n_universities"
    if railway == "station":      return "n_train_stations"
    if aeroway in {"aerodrome", "terminal"}: return "n_airports"
    if shop == "mall":            return "n_malls"
    if pd.notna(shop) and shop != "mall": return "n_shops"
    if amenity in {"restaurant", "fast_food", "cafe"}: return "n_restaurants"
    if pd.notna(office):          return "n_offices"
    if tourism in {"hotel", "museum", "attraction", "guest_house", "hostel"}: return "n_tourism"

    return None


def assign_poi_layers(
    gdf_h3: gpd.GeoDataFrame,
    polys: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
    MIN_SHARE_AREA: float = 0.05,
    native_crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Assign POI-based functional layer counts to each H3 hexagon.

    For each hexagon, counts the number of POIs in each functional category
    (schools, hospitals, shops, etc.) via a spatial join. Parks are handled
    separately using polygon area overlap rather than point counts, since a
    single large park polygon can cover multiple hexagons.

    Parameters
    ----------
    gdf_h3         : H3 hexagon GeoDataFrame with column 'h3_index'
    polys          : output of download_and_prepare_landuse_polys()
    pois           : output of download_and_prepare_pois()
    MIN_SHARE_AREA : minimum fraction of a hex's area that a park polygon must
                     cover for the park to be counted in that hex (default 0.05)
    native_crs     : output CRS (default EPSG:4326)

    Returns
    -------
    gpd.GeoDataFrame with new integer columns:
    n_schools, n_hospitals, n_parks, n_train_stations, n_airports,
    n_malls, n_restaurants, n_shops, n_offices, n_universities, n_tourism
    """
    # Reproject to metric CRS for consistent spatial joins
    crs    = polys.estimate_utm_crs()
    gdf_h3 = gdf_h3.to_crs(crs)
    polys  = polys.to_crs(crs)
    pois   = pois.to_crs(crs)

    # Classify each POI and drop unrecognised ones
    pois["poi_col"] = pois.apply(classify_poi, axis=1)
    pois_use = pois[pois["poi_col"].notna()].copy()

    # Spatial join: count POIs within each hexagon
    poi_join = gpd.sjoin(
        pois_use[["poi_col", "geometry"]],
        gdf_h3[["h3_index", "geometry"]],
        how="inner",
        predicate="within"
    ).drop(columns="index_right", errors="ignore")

    poi_counts = (
        poi_join.groupby(["h3_index", "poi_col"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # --- Parks: area-based rather than point-based ---
    # Parks are polygon features in OSM; a point centroid would misrepresent
    # large parks that span multiple hexagons. Instead, count a park as present
    # in a hex only if its intersection covers at least MIN_SHARE_AREA of the hex.
    park_polys = polys[
        polys.get("leisure", pd.Series(index=polys.index)).isin(["park", "garden"]) &
        polys.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    if not park_polys.empty:
        park_polys = park_polys.reset_index(drop=True).copy()
        park_polys["park_id"] = park_polys.index

        park_hit = gpd.overlay(
            gdf_h3[["h3_index", "geometry"]],
            park_polys[["park_id", "geometry"]],
            how="intersection"
        )

        if not park_hit.empty:
            hex_area = gdf_h3.set_index("h3_index").geometry.area
            park_hit["int_area"] = park_hit.geometry.area
            park_hit["hex_area"] = park_hit["h3_index"].map(hex_area)
            park_hit["share"]    = park_hit["int_area"] / park_hit["hex_area"]

            park_hit = park_hit[park_hit["share"] >= MIN_SHARE_AREA].copy()

            park_counts = (
                park_hit.groupby("h3_index")["park_id"]
                .nunique()
                .rename("n_parks")
                .reset_index()
            )
        else:
            park_counts = pd.DataFrame(columns=["h3_index", "n_parks"])
    else:
        park_counts = pd.DataFrame(columns=["h3_index", "n_parks"])

    # --- Merge all POI counts into the hexagon layer ---
    cols = [
        "n_schools", "n_hospitals", "n_parks", "n_train_stations",
        "n_airports", "n_malls", "n_restaurants", "n_shops",
        "n_offices", "n_universities", "n_tourism",
    ]

    gdf_h3 = gdf_h3.merge(poi_counts,  on="h3_index", how="left")
    gdf_h3 = gdf_h3.merge(park_counts, on="h3_index", how="left")

    # Fill missing columns with 0 and cast to int
    for c in cols:
        if c not in gdf_h3.columns:
            gdf_h3[c] = 0
    gdf_h3[cols] = gdf_h3[cols].fillna(0).astype(int)

    gdf_h3 = gdf_h3.to_crs(native_crs)
    return gdf_h3


# ---------------------------------------------------------------------------
# Highway layer assignment
# ---------------------------------------------------------------------------

def assign_highway_layer(
    osm,
    gdf_h3: gpd.GeoDataFrame,
    filters: list = ["motorway", "trunk"],
    native_crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Add a binary 'is_highway' flag to each H3 hexagon.

    Extracts the driving network from the PBF file, filters to the specified
    highway types (default: motorway and trunk), and marks hexagons that
    intersect at least one matching road segment with is_highway=1.

    Parameters
    ----------
    osm        : pyrosm.OSM  — OSM reader initialised on the PBF file
    gdf_h3     : H3 hexagon GeoDataFrame with column 'h3_index'
    filters    : list of OSM highway tag values to include
    native_crs : output CRS (default EPSG:4326)

    Returns
    -------
    gpd.GeoDataFrame with a new column 'is_highway' (0 or 1)
    """
    print("Reading highway network from PBF...")

    edges = osm.get_network(network_type="driving", extra_attributes=["highway"])

    if filters:
        edges = edges[edges["highway"].isin(filters)].copy()

    edges = edges.to_crs(native_crs)

    # Spatial join: flag hexagons that intersect any highway segment
    joined = gpd.sjoin(gdf_h3, edges[["geometry"]], predicate="intersects", how="inner")
    idx_intersecting = joined.index.unique()

    gdf_h3["is_highway"] = 0
    gdf_h3.loc[idx_intersecting, "is_highway"] = 1

    print(f"Hexagons with highway: {gdf_h3['is_highway'].sum():,} / {len(gdf_h3):,}")
    return gdf_h3
