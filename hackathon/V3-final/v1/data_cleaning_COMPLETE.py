#!/usr/bin/env python3
"""
Vehicle Sales — COMPLETE Data Cleaning Pipeline
================================================
Addresses ALL data quality issues found in forensic audit:
1. Column-shifted rows (26) - "sedan" in Transmission
2. Concatenated make+body (19) - "ford truck", "gmc truck", etc.
3. Make abbreviations (25) - "vw", "dot"
4. Split model names (88) - "range" + "rover sport"
5. Placeholder values - "—", 999999
6. Casing inconsistencies - lowercase makes
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

RAW_PATH = "hackathon/V2/data/VehicleSales.parquet"
OUT_PATH = "vehicle_sales_final.parquet"

df = pd.read_parquet(RAW_PATH)
print(f"Raw: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ═══════════════════════════════════════════════════════════════════
# FIX 1: Column-shifted rows (26 rows)
# "sedan"/"Sedan" in Transmission = values shifted right from Body column
# ═══════════════════════════════════════════════════════════════════
shift_mask = df["Transmission"].isin(["sedan", "Sedan"])
if shift_mask.sum() > 0:
    print(f"  Fixing {shift_mask.sum()} column-shifted rows...")
    df.loc[shift_mask, "Body"] = "Sedan"
    df.loc[shift_mask, "Transmission"] = df.loc[shift_mask, "VIN"]
    df.loc[shift_mask, "VIN"] = df.loc[shift_mask, "State"]
    df.loc[shift_mask, "State"] = np.nan

# ═══════════════════════════════════════════════════════════════════
# FIX 2: Concatenated make + body type (19 rows)
# "ford truck", "gmc truck", "dodge tk" etc.
# ═══════════════════════════════════════════════════════════════════
concat_mask = df["Make"].str.contains(r" truck$| tk$", case=False, na=False)
if concat_mask.sum() > 0:
    print(f"  Fixing {concat_mask.sum()} concatenated make+body rows...")
    # Extract body type suffix
    df.loc[concat_mask, "Body"] = df.loc[concat_mask, "Make"].str.extract(r"(truck|tk)$", expand=False)
    # Map tk → Truck
    df.loc[df["Body"] == "tk", "Body"] = "Truck"
    df.loc[df["Body"] == "truck", "Body"] = "Truck"
    # Clean the make
    df.loc[concat_mask, "Make"] = df.loc[concat_mask, "Make"].str.replace(r" (truck|tk)$", "", regex=True, case=False)

# ═══════════════════════════════════════════════════════════════════
# FIX 3: Make abbreviations and errors
# ═══════════════════════════════════════════════════════════════════
MAKE_FIXES = {
    "vw": "Volkswagen",
    "dot": "Dodge",
    "mercedes-b": "Mercedes-Benz",
}
for bad, good in MAKE_FIXES.items():
    mask = df["Make"].str.lower() == bad
    if mask.sum() > 0:
        print(f"  Fixing {mask.sum()} rows: '{bad}' → '{good}'")
        df.loc[mask, "Make"] = good

# ═══════════════════════════════════════════════════════════════════
# FIX 4: Split model names (Land Rover)
# Model="range", Trim="rover sport" → Model="Range Rover", Trim="Sport"
# ═══════════════════════════════════════════════════════════════════
range_mask = df["Model"].str.lower() == "range"
if range_mask.sum() > 0:
    print(f"  Fixing {range_mask.sum()} split 'Range Rover' rows...")
    # Combine: Trim was split, containing "rover sport", "rover hse", etc.
    df.loc[range_mask, "Model"] = "Range Rover"
    df.loc[range_mask, "Trim"] = df.loc[range_mask, "Trim"].str.replace(r"^rover\\s*", "", regex=True, case=False).str.strip()

# Model abbreviations
MODEL_FIXES = {
    "rangerover": "Range Rover",
    "rr": "Range Rover",
    "rrs": "Range Rover Sport",
    "lr3": "LR3",
}
for bad, good in MODEL_FIXES.items():
    mask = df["Model"].str.lower() == bad
    if mask.sum() > 0:
        print(f"  Fixing {mask.sum()} rows: Model '{bad}' → '{good}'")
        df.loc[mask, "Model"] = good

# ═══════════════════════════════════════════════════════════════════
# FIX 5: Placeholder values
# ═══════════════════════════════════════════════════════════════════
# Replace "—" em dash with NaN
em_dash_count = (df == "—").sum().sum()
if em_dash_count > 0:
    print(f"  Replacing {em_dash_count:,} '—' placeholders with NaN...")
    df = df.replace("—", np.nan)

# Replace 999999 odometer with NaN
odo_placeholder = (df["Odometer"] == 999999).sum()
if odo_placeholder > 0:
    print(f"  Replacing {odo_placeholder} odometer=999999 with NaN...")
    df.loc[df["Odometer"] == 999999, "Odometer"] = np.nan

# ═══════════════════════════════════════════════════════════════════
# STANDARD CLEANING (your existing pipeline)
# ═══════════════════════════════════════════════════════════════════
print("  Applying standard cleaning...")

# Parse dates
df["SaleDate"] = pd.to_datetime(df["SaleDate"], errors="coerce")

# Drop rows with critical nulls
df.dropna(subset=["SaleDate", "SellingPrice", "MMR"], inplace=True)

# Remove extreme price outliers (1st-99th percentile)
Q1, Q3 = df["SellingPrice"].quantile([0.01, 0.99])
df = df[(df["SellingPrice"] >= Q1) & (df["SellingPrice"] <= Q3)]

# Drop MMR nulls
df.dropna(subset=["MMR"], inplace=True)

# Standardize casing
for col in ["Make", "Model", "Body", "Color", "State"]:
    df[col] = df[col].str.strip().str.title()
df["Transmission"] = df["Transmission"].str.strip().str.lower()

# ═══════════════════════════════════════════════════════════════════
# ENGINEERED FEATURES
# ═══════════════════════════════════════════════════════════════════
df["SaleYear"] = df["SaleDate"].dt.year
df["SaleMonth"] = df["SaleDate"].dt.to_period("M").astype(str)
df["SaleQuarter"] = df["SaleDate"].dt.to_period("Q").astype(str)
df["VehicleAge"] = df["SaleYear"] - df["Year"]
df["PriceDiff"] = df["SellingPrice"] - df["MMR"]
df["PriceDiffPct"] = (df["PriceDiff"] / df["MMR"]) * 100
df["AboveMMR"] = df["PriceDiff"] > 0

bins = [0, 5000, 10000, 15000, 20000, 30000, 50000, 999999]
labels = ["<$5k", "$5-10k", "$10-15k", "$15-20k", "$20-30k", "$30-50k", ">$50k"]
df["PriceBand"] = pd.cut(df["SellingPrice"], bins=bins, labels=labels)

# ═══════════════════════════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"Cleaned: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"{'='*55}")

print(f"\nVerification (should all be 0 or false):")
print(f"  'Navitgation' in Body:         {(df['Body'] == 'Navitgation').sum()}")
print(f"  'sedan' in Transmission:        {df['Transmission'].isin(['sedan', 'Sedan']).sum()}")
print(f"  Make contains 'truck'/'tk':     {df['Make'].str.contains(r'truck| tk$', case=False, na=False).sum()}")
print(f"  Make='vw' or 'dot':             {(df['Make'].str.lower().isin(['vw', 'dot'])).sum()}")
print(f"  Model='range':                  {(df['Model'].str.lower() == 'range').sum()}")
print(f"  '—' em dash remaining:          {(df == '—').sum().sum()}")
print(f"  Odometer=999999:                {(df['Odometer'] == 999999).sum()}")

print(f"\nTransmission distribution:")
print(df["Transmission"].value_counts(dropna=False))

print(f"\nBody distribution (top 10):")
print(df["Body"].value_counts().head(10))

# Save
df.to_parquet(OUT_PATH, index=False)
print(f"\nSaved to: {OUT_PATH}")
