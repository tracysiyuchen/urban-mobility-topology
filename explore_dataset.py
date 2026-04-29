"""Quick EDA script for NYC Taxi dataset to inform model design decisions."""
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/nyc-taxi-trip-duration/train.csv"

print("=" * 60)
print("LOADING DATA")
print("=" * 60)
df = pd.read_csv(DATA_PATH, parse_dates=["pickup_datetime", "dropoff_datetime"])
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nNull counts:\n{df.isnull().sum()}")

print("\n" + "=" * 60)
print("TEMPORAL RANGE")
print("=" * 60)
print(f"Date range: {df['pickup_datetime'].min()} → {df['pickup_datetime'].max()}")
df["month"] = df["pickup_datetime"].dt.month
df["hour"] = df["pickup_datetime"].dt.hour
df["dow"] = df["pickup_datetime"].dt.dayofweek
print(f"\nTrips per month:\n{df['month'].value_counts().sort_index()}")
print(f"\nTrips per hour (sample):\n{df['hour'].value_counts().sort_index()}")

# Time bins
def time_bin(hour):
    if 6 <= hour < 10:   return "Morning Peak"
    elif 10 <= hour < 16: return "Midday"
    elif 16 <= hour < 20: return "Evening Peak"
    else:                 return "Night"

df["time_bin"] = df["hour"].apply(time_bin)
print(f"\nTrips per time bin:\n{df['time_bin'].value_counts()}")

print("\n" + "=" * 60)
print("SPATIAL EXTENT")
print("=" * 60)
for col in ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]:
    print(f"{col}: [{df[col].min():.4f}, {df[col].max():.4f}], mean={df[col].mean():.4f}")

# NYC bounding box: lon [-74.25, -73.70], lat [40.50, 40.92]
nyc_lon = (-74.25, -73.70)
nyc_lat = (40.50, 40.92)
in_bbox = (
    df["pickup_longitude"].between(*nyc_lon) &
    df["pickup_latitude"].between(*nyc_lat) &
    df["dropoff_longitude"].between(*nyc_lon) &
    df["dropoff_latitude"].between(*nyc_lat)
)
print(f"\nTrips with both endpoints in NYC bbox: {in_bbox.sum()} ({in_bbox.mean()*100:.2f}%)")
print(f"Trips outside bbox (to filter): {(~in_bbox).sum()}")

df_nyc = df[in_bbox].copy()

print("\n" + "=" * 60)
print("TRIP DURATION STATS (seconds)")
print("=" * 60)
print(df_nyc["trip_duration"].describe())
# Filter extreme durations: keep 1 min < duration < 3 hours
valid_dur = df_nyc["trip_duration"].between(60, 10800)
print(f"\nAfter filtering (1min–3hr): {valid_dur.sum()} trips ({valid_dur.mean()*100:.2f}%)")

print("\n" + "=" * 60)
print("H3 CELL ANALYSIS (requires h3 package)")
print("=" * 60)
try:
    import h3
    df_clean = df_nyc[valid_dur].copy()

    for res in [6, 7, 8]:
        df_clean[f"pickup_h3_{res}"] = df_clean.apply(
            lambda r: h3.latlng_to_cell(r["pickup_latitude"], r["pickup_longitude"], res), axis=1
        )
        df_clean[f"dropoff_h3_{res}"] = df_clean.apply(
            lambda r: h3.latlng_to_cell(r["dropoff_latitude"], r["dropoff_longitude"], res), axis=1
        )
        n_pickup_cells = df_clean[f"pickup_h3_{res}"].nunique()
        n_dropoff_cells = df_clean[f"dropoff_h3_{res}"].nunique()
        all_cells = set(df_clean[f"pickup_h3_{res}"]) | set(df_clean[f"dropoff_h3_{res}"])
        print(f"\nH3 resolution {res}:")
        print(f"  Unique pickup cells: {n_pickup_cells}")
        print(f"  Unique dropoff cells: {n_dropoff_cells}")
        print(f"  Total unique cells (union): {len(all_cells)}")

        # OD matrix sparsity per time bin
        for tbin in ["Morning Peak", "Midday", "Evening Peak", "Night"]:
            subset = df_clean[df_clean["time_bin"] == tbin]
            od_pairs = subset.groupby([f"pickup_h3_{res}", f"dropoff_h3_{res}"]).size()
            n_cells = len(all_cells)
            total_possible = n_cells * n_cells
            density = len(od_pairs) / total_possible
            print(f"  [{tbin}] OD pairs: {len(od_pairs)}, sparsity: {(1-density)*100:.2f}%")

    # Per-date OD snapshot stats
    res = 7
    df_clean["date"] = df_clean["pickup_datetime"].dt.date
    snapshots = df_clean.groupby(["date", "time_bin"]).size()
    print(f"\nTotal (date, time_bin) snapshots: {len(snapshots)}")
    print(f"Trips per snapshot: min={snapshots.min()}, median={snapshots.median():.0f}, max={snapshots.max()}")

    # Cell-level statistics
    cells_r7 = set(df_clean["pickup_h3_7"]) | set(df_clean["dropoff_h3_7"])
    print(f"\nAt H3 res=7, total active cells: {len(cells_r7)}")

    # Trips per cell (outflow)
    outflow = df_clean["pickup_h3_7"].value_counts()
    print(f"Outflow per cell: mean={outflow.mean():.1f}, median={outflow.median():.1f}, max={outflow.max()}")
    print(f"Cells with < 100 trips: {(outflow < 100).sum()} ({(outflow < 100).mean()*100:.1f}%)")

except ImportError:
    print("h3 not installed. Install with: pip install h3")

print("\n" + "=" * 60)
print("VENDOR & PASSENGER STATS")
print("=" * 60)
print(f"Vendor distribution:\n{df['vendor_id'].value_counts()}")
print(f"\nPassenger count distribution:\n{df['passenger_count'].value_counts().sort_index()}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
