#!/usr/bin/env python3
"""
Vehicle Data Cleaning - FAST VERSION (Skips Slow KNN)
======================================================
This version completes in ~10 seconds by skipping the slow KNN imputation.
All structural fixes are applied. Missing Color/Interior can be imputed later if needed.
"""

import pandas as pd
import numpy as np
import warnings
import time
import sys

warnings.filterwarnings('ignore')
np.random.seed(42)

# Timing helper
class Timer:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self.t = time.time()
        return self
    def __exit__(self, *args):
        print(f'  ✓ {self.label}: {time.time()-self.t:.2f}s')

audit_log = []

def log(msg):
    audit_log.append(msg)
    print(f'  [{msg[:6]}] {msg}')

print('='*70)
print('VEHICLE DATA FORENSIC CLEANING - FAST VERSION')
print('='*70)

# Configuration
INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else '/home/mennatullah/Documents/repos/li/AI/instant/hackathon/data/VehicleSales.parquet'
OUTPUT_FILE = sys.argv[2] if len(sys.argv) > 2 else 'VehicleSales_CLEANED.parquet'

print(f'Input:  {INPUT_FILE}')
print(f'Output: {OUTPUT_FILE}')
print('='*70)

# Load data
print('\n[1/7] Loading data...')
with Timer('Load parquet'):
    try:
        df = pd.read_parquet(INPUT_FILE)
        print(f'  Loaded {df.shape[0]:,} rows x {df.shape[1]} columns')
        null_before = df.isnull().sum()
    except Exception as e:
        print(f'✗ Error: {e}')
        sys.exit(1)

# Phase 1: Structural Fixes (Column Shifts)
print('\n[2/7] Phase 1: Structural fixes...')

# CS-001: Column-shifted rows
with Timer('CS-001 column shift'):
    shift_mask = df['Transmission'].isin(['sedan', 'Sedan'])
    count = shift_mask.sum()
    if count > 0:
        orig = df.loc[shift_mask, ['Transmission', 'VIN', 'State']].copy()
        df.loc[shift_mask, 'State'] = np.nan
        df.loc[shift_mask, 'VIN'] = orig['State'].values
        df.loc[shift_mask, 'Transmission'] = orig['VIN'].values
        df.loc[shift_mask, 'Body'] = 'Sedan'
        df.loc[shift_mask, 'ConditionValue'] = np.nan
        log(f'CS-001: Fixed {count} column-shifted rows')
    else:
        print('  No column shifts detected')

# CS-002: Make-body concatenations (vectorized)
with Timer('CS-002 make-body concat'):
    make_body_map = {
        'ford truck': ('Ford', 'Truck'), 'gmc truck': ('GMC', 'Truck'),
        'chev truck': ('Chevrolet', 'Truck'), 'chev tk': ('Chevrolet', 'Truck'),
        'ford tk': ('Ford', 'Truck'), 'dodge tk': ('Dodge', 'Truck'),
        'mazda tk': ('Mazda', 'Truck'), 'hyundai tk': ('Hyundai', 'Truck'),
    }
    make_lower = df['Make'].str.lower()
    for raw, (mk, bd) in make_body_map.items():
        mask = make_lower == raw
        if mask.sum() > 0:
            df.loc[mask, 'Make'] = mk
            df.loc[mask, 'Body'] = bd
            log(f'CS-002: "{raw}" → {mk}/{bd} ({mask.sum()} rows)')

# MF-001: Range Rover fragmentation (vectorized)
with Timer('MF-001 Range Rover fragmentation'):
    model_lower = df['Model'].str.lower()
    m1 = (model_lower == 'range') & df['Trim'].str.contains('rover', case=False, na=False)
    m2 = model_lower == 'rangerover'
    m3 = df['Model'].isin(['rr','RR','rrs','RRS'])
    rr_mask = m1 | m2 | m3
    if rr_mask.sum() > 0:
        df.loc[rr_mask, 'Model'] = 'Range Rover'
        log(f'MF-001: Fixed {rr_mask.sum()} Range Rover fragmentation rows')

# Phase 2: Standardization
print('\n[3/7] Phase 2: Standardization...')

with Timer('MK-001 make standardization'):
    make_mapping = {
        'vw': 'Volkswagen', 'dot': 'Dodge', 'mercedes-b': 'Mercedes-Benz',
        'mercedes-benz': 'Mercedes-Benz', 'chev': 'Chevrolet', 'chevy': 'Chevrolet'
    }
    df['Make'] = df['Make'].str.lower().replace(make_mapping).str.title()
    log('MK-001: Make names standardized')

with Timer('Categorical standardization'):
    trans_map = {'auto': 'automatic', 'man': 'manual'}
    df['Transmission'] = df['Transmission'].str.strip().str.lower().replace(trans_map)
    for col in ['Make', 'Model', 'Body', 'Color', 'Interior', 'State', 'Trim']:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()
    log('STD: Categoricals standardized')

# Phase 3: Sentinel Values → NaN (MNAR)
print('\n[4/7] Phase 3: Converting MNAR sentinels...')
with Timer('PH sentinel conversion'):
    m = df['Odometer'] == 999999
    if m.sum() > 0:
        df.loc[m, 'Odometer'] = np.nan
        log(f'PH-001: {m.sum()} odometer sentinels → NaN')
    for col in ['Color', 'Interior']:
        m = df[col] == '---'
        if m.sum() > 0:
            df.loc[m, col] = np.nan
            log(f'PH-002: {m.sum()} {col} placeholders → NaN')
    if 'MMR' in df.columns:
        m = df['MMR'] == 999999
        if m.sum() > 0:
            df.loc[m, 'MMR'] = np.nan
            log(f'PH-003: {m.sum()} MMR sentinels → NaN')

# Phase 4: VIN-Based Imputation (Domain Knowledge)
print('\n[5/7] Phase 4: VIN-based imputation...')

WMI_DB = {
    '1FT':('Ford','Truck'), '1G1':('Chevrolet',None), '1GC':('Chevrolet','Truck'),
    '1GT':('GMC','Truck'), '1HG':('Honda',None), '1J4':('Jeep',None),
    '1J8':('Jeep',None), '1N4':('Nissan',None), '1N6':('Nissan','Truck'),
    '1VW':('Volkswagen',None), '1YV':('Mazda',None), '19U':('Acura',None),
    '2G1':('Chevrolet',None), '2T1':('Toyota',None), '2T3':('Toyota','Suv'),
    '3VW':('Volkswagen',None), '4S3':('Subaru',None), '4S4':('Subaru',None),
    '5J6':('Honda','Suv'), '5N1':('Hyundai','Suv'), '5UX':('BMW',None),
    '5XX':('Kia',None), 'JM1':('Mazda',None), 'JTD':('Toyota',None),
    'JTH':('Lexus',None), 'JTJ':('Lexus','Suv'), 'KNA':('Kia',None),
    'KNB':('Kia',None), 'KND':('Kia',None), 'KMH':('Hyundai',None),
    'SAL':('Land Rover',None), 'SAJ':('Jaguar',None), 'SCA':('Rolls Royce',None),
    'SCB':('Bentley',None), 'SCF':('Aston Martin',None), 'WA1':('Audi','Suv'),
    'WAU':('Audi',None), 'WBA':('BMW',None), 'WBS':('BMW',None),
    'WBX':('BMW','Suv'), 'WDC':('Mercedes-Benz','Suv'), 'WDD':('Mercedes-Benz',None),
    'WP0':('Porsche',None), 'WP1':('Porsche','Suv'), 'WVW':('Volkswagen',None),
    'YV1':('Volvo',None), 'YV4':('Volvo','Suv'), 'ZFF':('Ferrari',None),
    '1FA':('Ford',None), '1FM':('Ford','Suv'), '1LN':('Lincoln',None),
    '1B4':('Dodge',None), '1C3':('Chrysler',None), '1C4':('Chrysler',None),
    '3N1':('Nissan',None), '4T1':('Toyota',None), '2HG':('Honda',None),
    '2HK':('Honda','Suv'), 'JN1':('Nissan',None), 'JF1':('Subaru',None),
    'JF2':('Subaru',None), 'JH4':('Acura',None), '1GK':('GMC','Suv'),
    'ZAM':('Maserati',None), 'ZAR':('Alfa Romeo',None), 'ZFA':('Fiat',None),
}

with Timer('IMP-VIN vectorized'):
    missing_make = df['Make'].isna() & df['VIN'].notna()
    if missing_make.sum() > 0:
        wmi_series = df.loc[missing_make, 'VIN'].str[:3].str.upper()
        make_from_wmi = wmi_series.map({k: v[0] for k, v in WMI_DB.items()})
        body_from_wmi = wmi_series.map({k: v[1] for k, v in WMI_DB.items()})
        n_make = make_from_wmi.notna().sum()
        df.loc[missing_make, 'Make'] = df.loc[missing_make, 'Make'].fillna(make_from_wmi)
        missing_both = missing_make & df['Body'].isna()
        if missing_both.sum() > 0:
            wmi_body = df.loc[missing_both, 'VIN'].str[:3].str.upper().map({k: v[1] for k, v in WMI_DB.items()})
            df.loc[missing_both, 'Body'] = df.loc[missing_both, 'Body'].fillna(wmi_body)
        log(f'IMP-VIN: Imputed {n_make} Make values from WMI')

# Phase 5: Simple Mode Imputation for Color/Interior (FAST alternative to KNN)
print('\n[6/7] Phase 5: Fast mode imputation for Color/Interior...')
with Timer('Mode imputation'):
    # Simple mode imputation by Make+Body - much faster than KNN
    for col in ['Color', 'Interior']:
        if col in df.columns:
            missing = df[col].isna()
            if missing.sum() > 0:
                # Group by Make and Body, fill with mode
                for (make, body), group in df.groupby(['Make', 'Body']):
                    mode_val = group[col].mode()
                    if not mode_val.empty:
                        mask = missing & (df['Make'] == make) & (df['Body'] == body)
                        if mask.sum() > 0:
                            df.loc[mask, col] = mode_val[0]

                # Fill any remaining with overall mode
                still_missing = df[col].isna()
                if still_missing.sum() > 0:
                    overall_mode = df[col].mode()
                    if not overall_mode.empty:
                        df.loc[still_missing, col] = overall_mode[0]

                filled = missing.sum() - df[col].isna().sum()
                log(f'IMP-MODE: {col} - {filled} values imputed by mode')

# Phase 6: Flag Non-Imputable
print('\n[7/7] Phase 7: Flagging non-imputable rows...')
with Timer('FLAG non-imputable'):
    cat_cols = [c for c in ['Make','Model','Trim','Body'] if c in df.columns]
    all_null = df[cat_cols].isna().all(axis=1)
    no_vin = df['VIN'].isna() | (df['VIN'].astype(str).str.strip() == '')
    non_imp = all_null & no_vin
    df['_flag_non_imputable'] = non_imp
    log(f'FLAG: {non_imp.sum():,} rows flagged as non-imputable')

# Summary
print('\n' + '='*70)
print('CLEANING COMPLETE')
print('='*70)

null_after = df.isnull().sum()
comparison = pd.DataFrame({
    'Null Before': null_before,
    'Null After': null_after,
    'Fixed': null_before - null_after
})
comparison = comparison[comparison['Null Before'] > 0].sort_values('Fixed', ascending=False)
print('\n=== NULL COMPARISON ===')
print(comparison.to_string())

print(f'\nFinal shape: {df.shape[0]:,} rows x {df.shape[1]} columns')
print(f'Non-imputable rows: {df["_flag_non_imputable"].sum():,}')
print('\n=== AUDIT LOG ===')
for entry in audit_log:
    print(f'  • {entry}')

# Save
print(f'\nSaving to {OUTPUT_FILE}...')
try:
    if OUTPUT_FILE.endswith('.parquet'):
        df.to_parquet(OUTPUT_FILE, index=False)
    else:
        df.to_csv(OUTPUT_FILE, index=False)
    print(f'✓ Saved successfully')
    print(f'  File size: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB')
except Exception as e:
    print(f'✗ Error saving: {e}')
    sys.exit(1)

print('='*70)
print('NOTE: This fast version uses mode imputation instead of KNN.')
print('KNN imputation can be added later if variance preservation is critical.')
print('='*70)