"""
rollup_utils.py
------------
Utility functions for computing 1-row daily rollup summaries from the
per-user and per-hex statistics produced by compute_daily_stats.py.
Outputs feed directly into the EDA+QA notebooks for time-series analysis.
"""


import pandas as pd
import numpy as np


def compute_user_daily_rollup(user_daily_stats_day: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a 1-row daily rollup from a single-day user_daily_stats table.
 
    Parameters
    ----------
    user_daily_stats_day : pd.DataFrame
        Single-day output of the user aggregation in compute_daily_stats.py.
        Required columns: ['date', 'uid', 'n_points', 'n_hexes', 'active_window_m'].
 
    Returns
    -------
    pd.DataFrame
        One row with the following column groups:
        - Coverage      : n_users
        - Points        : n_users_le_{1,2,3,5,10}, n_points, avg/p50/p75/p90/p99_points
        - Hexes         : n_users_hexes_le{1,2}, avg/p50/p75/p90_hexes
        - Active window : avg/p50/p75_active_win (minutes)
    """


    date_vals = user_daily_stats_day["date"].dropna().unique()
    if len(date_vals) != 1:
        raise ValueError(f"Expected a single date in the input, got {len(date_vals)}: {date_vals[:5]}")
    date = date_vals[0]
    
    # Coerce to numeric (safe against mixed-type parquet reads)
    n_points = pd.to_numeric(user_daily_stats_day["n_points"], errors="coerce")
    n_hexes = pd.to_numeric(user_daily_stats_day["n_hexes"], errors="coerce")
    active_window = pd.to_numeric(user_daily_stats_day["active_window_m"], errors="coerce")
    
    n_users = user_daily_stats_day["uid"].nunique(dropna=True)
    
    # Left-tail counts: share of users with very sparse daily traces
    n_users_le_1 = int((n_points <= 1).sum(skipna=True))
    n_users_le_2 = int((n_points <= 2).sum(skipna=True))
    n_users_le_3 = int((n_points <= 3).sum(skipna=True))
    n_users_le_5 = int((n_points <= 5).sum(skipna=True))
    n_users_le_10 = int((n_points <= 10).sum(skipna=True))
    
    # Points distribution
    tot_points = float(n_points.sum())
    avg_points = float(n_points.mean())
    p50_points = float(n_points.quantile(0.50))
    p75_points = float(n_points.quantile(0.75))
    p90_points = float(n_points.quantile(0.90))
    p99_points = float(n_points.quantile(0.99))

    # Hexes distribution
    avg_hexes = float(n_hexes.mean())
    p50_hexes = float(n_hexes.quantile(0.50))
    p75_hexes = float(n_hexes.quantile(0.75))
    p90_hexes = float(n_hexes.quantile(0.90))

    # Left-tail counts: spatially localised users
    n_users_hexes_le1 = int((n_hexes <= 1).sum(skipna=True))
    n_users_hexes_le2 = int((n_hexes <= 2).sum(skipna=True))

    # Active window distribution
    avg_active_win = float(active_window.mean())
    p50_active_win = float(active_window.quantile(0.50))
    p75_active_win = float(active_window.quantile(0.75))
    
    
    out = pd.DataFrame([{
        "date": date,
        "n_users": int(n_users),
        "n_users_le_1": n_users_le_1,
        "n_users_le_2": n_users_le_2,
        "n_users_le_3": n_users_le_3,
        "n_users_le_5": n_users_le_5,
        "n_users_le_10": n_users_le_10,
    
        "n_points": tot_points,
        "avg_points": avg_points,
        "p50_points": p50_points,
        "p75_points": p75_points,
        "p90_points": p90_points,      
        "p99_points": p99_points,

        "avg_hexes": avg_hexes,
        "p50_hexes": p50_hexes,
        "p75_hexes": p75_hexes,
        "p90_hexes": p90_hexes,

        "n_users_hexes_le1": n_users_hexes_le1,
        "n_users_hexes_le2": n_users_hexes_le2,
    
        "avg_active_win": avg_active_win,
        "p50_active_win": p50_active_win,
        "p75_active_win": p75_active_win
    
    }])
    
    return out


def compute_hex_daily_rollup(hex_daily_stats_day: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a 1-row daily rollup from a single-day hex_daily_stats table.
 
    Parameters
    ----------
    hex_daily_stats_day : pd.DataFrame
        Single-day output of the spatial aggregation in compute_daily_stats.py.
        Required columns: ['date', 'hex_id', 'n_points', 'n_users'].
 
    Returns
    -------
    pd.DataFrame
        One row with the following column groups:
        - Coverage       : n_hexes_active, n_points
        - Low-activity   : n_hexes_points_le_{1,2,5}, n_hexes_users_le_{1,2,3}
        - Distributions  : p50/75/90/99_points_hex, p50/75/90/99_users_hex
        - Concentration  : top_{1,5,10}pct_share_points (share of total points
                           held by the top k% most visited hexagons)
    """

   
    date_vals = hex_daily_stats_day["date"].dropna().unique()
    if len(date_vals) != 1:
        raise ValueError(f"Expected a single date in the input, got {len(date_vals)}")
    date = date_vals[0]

    # fillna(0): inactive hexes are included in the denominator for
    # concentration metrics; NaNs would skew percentiles upward.
    n_points = pd.to_numeric(hex_daily_stats_day["n_points"], errors="coerce").fillna(0)
    n_users = pd.to_numeric(hex_daily_stats_day["n_users"], errors="coerce").fillna(0)
 
    # Active hexes
    active_mask = (n_points > 0)
    n_hexes_active = int(active_mask.sum())
 
    total_points = float(n_points.sum())
 
    # Low-activity diagnostics (points)
    n_hexes_points_le_1 = int((n_points <= 1).sum())
    n_hexes_points_le_2 = int((n_points <= 2).sum())
    n_hexes_points_le_5 = int((n_points <= 5).sum())
 
    # Low-activity diagnostics (users)
    n_hexes_users_le_1 = int((n_users <= 1).sum())
    n_hexes_users_le_2 = int((n_users <= 2).sum())
    n_hexes_users_le_3 = int((n_users <= 3).sum())
 
    # Per-hex distributions
    p50_points_hex = float(n_points.quantile(0.50))
    p75_points_hex = float(n_points.quantile(0.75))
    p90_points_hex = float(n_points.quantile(0.90))
    p99_points_hex = float(n_points.quantile(0.99))
 
    p50_users_hex = float(n_users.quantile(0.50))
    p75_users_hex = float(n_users.quantile(0.75))
    p90_users_hex = float(n_users.quantile(0.90))
    p99_users_hex = float(n_users.quantile(0.99))
 
    # Concentration metrics
    def top_share(values: pd.Series, pct: float) -> float:
        """Share of total_points held by the top `pct` fraction of hexes."""
        if total_points <= 0:
            return 0.0
        m = len(values)
        k = max(1, int(np.ceil(m * pct)))
        return float(values.nlargest(k).sum() / total_points)
 
    top_1pct_share_points = top_share(n_points, 0.01)
    top_5pct_share_points = top_share(n_points, 0.05)
    top_10pct_share_points = top_share(n_points, 0.10)
 
    # Diagnostic only (double-counts users across hexes)
    sum_hex_users = float(n_users.sum())
 
    out = pd.DataFrame([{
        "date": date,
        "n_hexes_active": n_hexes_active,
        "n_points": total_points,
        
        "n_hexes_points_le_1": n_hexes_points_le_1,
        "n_hexes_points_le_2": n_hexes_points_le_2,
        "n_hexes_points_le_5": n_hexes_points_le_5,
        
        "n_hexes_users_le_1": n_hexes_users_le_1,
        "n_hexes_users_le_2": n_hexes_users_le_2,
        "n_hexes_users_le_3": n_hexes_users_le_3,
        
        "p50_points_hex": p50_points_hex,
        "p75_points_hex": p75_points_hex,
        "p90_points_hex": p90_points_hex,
        "p99_points_hex": p99_points_hex,
        
        "p50_users_hex": p50_users_hex,
        "p75_users_hex": p75_users_hex,
        "p90_users_hex": p90_users_hex,
        "p99_users_hex": p99_users_hex,
        
        "top_1pct_share_points": top_1pct_share_points,
        "top_5pct_share_points": top_5pct_share_points,
        "top_10pct_share_points": top_10pct_share_points,
    }])
 
    return out