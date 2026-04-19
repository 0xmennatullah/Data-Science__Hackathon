#!/usr/bin/env python3
"""
Power BI Data Export Helper
===========================
Since Power BI Desktop is Windows-only and can't be auto-generated from Python,
this script exports your cleaned data to CSV/Excel formats that Power BI
can easily import. Also creates a data dictionary for your mentor.

Usage:
    python powerbi_export.py

Output:
    - vehicle_sales_for_powerbi.csv   (main dataset)
    - data_dictionary.md              (column descriptions for Power BI setup)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─── Load & clean data (same corrected pipeline) ────────────────────────────
RAW_PATH = "hackathon/V2/data/VehicleSales.parquet"  # update as needed

df = pd.read_parquet(RAW_PATH)

# Fix column-shifted rows
shift_mask = df["Transmission"].isin(["sedan", "Sedan"])
if shift_mask.sum():
    df.loc[shift_mask, "Body"] = "Sedan"
    df.loc[shift_mask, "Transmission"] = df.loc[shift_mask, "VIN"]
    df.loc[shift_mask, "VIN"] = df.loc[shift_mask, "State"]
    df.loc[shift_mask, "State"] = np.nan

# Standard cleaning
df["SaleDate"] = pd.to_datetime(df["SaleDate"], errors="coerce")
df.dropna(subset=["SaleDate", "SellingPrice", "MMR"], inplace=True)
Q1, Q3 = df["SellingPrice"].quantile([0.01, 0.99])
df = df[(df["SellingPrice"] >= Q1) & (df["SellingPrice"] <= Q3)]
df.dropna(subset=["MMR"], inplace=True)

for col in ["Make", "Model", "Body", "Color", "State"]:
    df[col] = df[col].str.strip().str.title()
df["Transmission"] = df["Transmission"].str.strip().str.lower()

# Features
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

# Select columns for Power BI (drop less useful ones)
output_cols = [
    "Id", "Year", "Make", "Model", "Trim", "Body", "Transmission",
    "State", "ConditionValue", "Odometer", "Color", "Interior",
    "MMR", "SellingPrice", "SaleDate", "SaleYear", "SaleMonth",
    "SaleQuarter", "VehicleAge", "PriceDiff", "PriceDiffPct",
    "AboveMMR", "PriceBand"
]

output_df = df[output_cols].copy()

# Export to CSV (Power BI works best with clean CSV)
csv_path = "vehicle_sales_for_powerbi.csv"
output_df.to_csv(csv_path, index=False)
print(f"Exported: {csv_path} ({len(output_df):,} rows)")

# Also export aggregated tables for common Power BI use cases
# Monthly summary
monthly = output_df.groupby("SaleMonth").agg(
    Volume=("Id", "count"),
    Total_Revenue=("SellingPrice", "sum"),
    Avg_Price=("SellingPrice", "mean"),
    Median_Price=("SellingPrice", "median"),
    Avg_MMR=("MMR", "mean"),
    Above_MMR_Count=("AboveMMR", "sum")
).reset_index()
monthly["Above_MMR_Pct"] = (monthly["Above_MMR_Count"] / monthly["Volume"] * 100).round(1)
monthly.to_csv("monthly_summary_for_powerbi.csv", index=False)
print(f"Exported: monthly_summary_for_powerbi.csv")

# Make summary
make_summary = output_df.groupby("Make").agg(
    Volume=("Id", "count"),
    Median_Price=("SellingPrice", "median"),
    Avg_Condition=("ConditionValue", "mean"),
    Avg_Age=("VehicleAge", "mean")
).reset_index()
make_summary.to_csv("make_summary_for_powerbi.csv", index=False)
print(f"Exported: make_summary_for_powerbi.csv")

# Data dictionary
dictionary = """# Vehicle Sales — Data Dictionary for Power BI

## Main Table: vehicle_sales_for_powerbi.csv

| Column | Type | Description |
|--------|------|-------------|
| Id | Integer | Unique record identifier |
| Year | Integer | Vehicle model year |
| Make | Text | Vehicle manufacturer (Ford, Chevrolet, etc.) |
| Model | Text | Vehicle model name |
| Trim | Text | Vehicle trim level |
| Body | Text | Body type (Sedan, SUV, Hatchback, etc.) |
| Transmission | Text | automatic or manual |
| State | Text | US state code (CA, FL, TX, etc.) |
| ConditionValue | Decimal | Auction condition score (1-50) |
| Odometer | Integer | Mileage at time of sale |
| Color | Text | Exterior color |
| Interior | Text | Interior color/material |
| MMR | Currency | Manheim Market Reference (wholesale market value) |
| SellingPrice | Currency | Actual auction sale price |
| SaleDate | DateTime | Auction sale date |
| SaleYear | Integer | Year of sale (2014-2015) |
| SaleMonth | Text | Month of sale (YYYY-MM format) |
| SaleQuarter | Text | Quarter of sale (YYYYQX format) |
| VehicleAge | Integer | Age at auction (SaleYear - Year) |
| PriceDiff | Currency | SellingPrice - MMR (positive = sold above market) |
| PriceDiffPct | Decimal | PriceDiff / MMR * 100 |
| AboveMMR | Boolean | True if sold above MMR |
| PriceBand | Text | Price segment (<$5k, $5-10k, etc.) |

## Aggregation Tables

- **monthly_summary_for_powerbi.csv** — Pre-aggregated monthly KPIs
- **make_summary_for_powerbi.csv** — Pre-aggregated manufacturer-level KPIs

## How to Import into Power BI

1. Open Power BI Desktop (Windows only)
2. Click **Get Data → Text/CSV**
3. Select `vehicle_sales_for_powerbi.csv`
4. Click **Load**
5. Create visuals by dragging fields to the canvas

## Quick Measures for Power BI (DAX)

```
// % Above MMR
DIVIDE(COUNTROWS(FILTER('vehicle_sales', [AboveMMR] = TRUE)), COUNTROWS('vehicle_sales')) * 100

// Avg Price Diff %
AVERAGE('vehicle_sales'[PriceDiffPct])

// Median Price
MEDIAN('vehicle_sales'[SellingPrice])
```
"""

with open("data_dictionary.md", "w") as f:
    f.write(dictionary)
print(f"Exported: data_dictionary.md")
print("\nDone! Give these files to your mentor for Power BI import.")
