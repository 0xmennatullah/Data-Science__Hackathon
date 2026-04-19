#!/usr/bin/env python3
"""
Vehicle Sales — Corrected Data Cleaning Pipeline
================================================
Fixes the column-shift issue in 26 rows where "sedan" appeared in the 
Transmission column. The original approach blindly replaced "sedan" with 
"automatic", leaving corrupt data (Body="Navitgation", VIN="automatic").

This script properly realigns the shifted columns.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─── Load raw data ──────────────────────────────────────────────────────────
RAW_PATH = "hackathon/V2/data/VehicleSales.parquet"  # adjust path as needed
df = pd.read_parquet(RAW_PATH)

print(f"Raw data: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ─── FIX 1: Column-shifted rows (the "sedan in Transmission" issue) ────────
# These 26 rows have data shifted right by 1 column starting from Body:
#   Body="Navitgation"      ← garbage from parsing error
#   Transmission="Sedan"    ← this is actually the BODY type
#   VIN="automatic"/None    ← this is actually the TRANSMISSION
#   State="3vwd17aj..."     ← this is actually the VIN
#   ConditionValue=NaN      ← original State is lost (unrecoverable)

shift_mask = df["Transmission"].isin(["sedan", "Sedan"])
shifted_count = shift_mask.sum()

if shifted_count > 0:
    print(f"\nFound {shifted_count} column-shifted rows. Realigning...")

    # Cascade values left: each column gets the value from the column to its right
    df.loc[shift_mask, "Body"] = "Sedan"  # Transmission held the body type
    df.loc[shift_mask, "Transmission"] = df.loc[shift_mask, "VIN"]  # VIN held transmission
    df.loc[shift_mask, "VIN"] = df.loc[shift_mask, "State"]  # State held the VIN
    df.loc[shift_mask, "State"] = np.nan  # Original state is unrecoverable

    print(f"  ✓ Body set to 'Sedan' for {shifted_count} rows")
    print(f"  ✓ Transmission recovered: {df.loc[shift_mask, 'Transmission'].value_counts().to_dict()}")
    print(f"  ✓ VIN recovered from State column")
    print(f"  ✓ State set to NaN (unrecoverable)")

# ─── FIX 2: Standardize casing ──────────────────────────────────────────────
for col in ["Make", "Model", "Body", "Color", "State"]:
    df[col] = df[col].str.strip().str.title()
df["Transmission"] = df["Transmission"].str.strip().str.lower()

# ─── FIX 3: Parse dates ────────────────────────────────────────────────────
df["SaleDate"] = pd.to_datetime(df["SaleDate"], errors="coerce")

# ─── FIX 4: Drop rows with critical nulls ──────────────────────────────────
df.dropna(subset=["SaleDate", "SellingPrice", "MMR"], inplace=True)

# ─── FIX 5: Remove extreme outliers ────────────────────────────────────────
Q1, Q3 = df["SellingPrice"].quantile([0.01, 0.99])
df = df[(df["SellingPrice"] >= Q1) & (df["SellingPrice"] <= Q3)]

# ─── FIX 6: Drop MMR nulls ─────────────────────────────────────────────────
df.dropna(subset=["MMR"], inplace=True)

# ─── Engineered features ───────────────────────────────────────────────────
df["SaleYear"] = df["SaleDate"].dt.year
df["SaleMonth"] = df["SaleDate"].dt.to_period("M").astype(str)
df["VehicleAge"] = df["SaleYear"] - df["Year"]
df["PriceDiff"] = df["SellingPrice"] - df["MMR"]
df["PriceDiffPct"] = (df["PriceDiff"] / df["MMR"]) * 100

bins = [0, 5000, 10000, 15000, 20000, 30000, 50000, 999999]
labels = ["<$5k", "$5-10k", "$10-15k", "$15-20k", "$20-30k", "$30-50k", ">$50k"]
df["PriceBand"] = pd.cut(df["SellingPrice"], bins=bins, labels=labels)

print(f"\nCleaned data: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Transmission distribution:")
print(df["Transmission"].value_counts(dropna=False))

# Save
output_path = "vehicle_sales_cleaned_corrected.parquet"
df.to_parquet(output_path, index=False)
print(f"\nSaved to: {output_path}")
