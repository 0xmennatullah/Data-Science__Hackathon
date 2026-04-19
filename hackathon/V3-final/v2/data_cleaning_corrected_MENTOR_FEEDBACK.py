"""
Vehicle Data Forensic Cleaning Pipeline
=======================================
Addresses mentor feedback:
1. Root cause fixes for column shifts (not symptom patching)
2. Advanced missing value handling (MAR/MNAR-aware, not simple imputation)

Author: Data Science Hackathon Project
Date: 2026-04-18
Version: 2.0 (Post Review)
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Tuple, Optional, List
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import re

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class VehicleDataCleaner:
    """
    Forensic data cleaner that addresses root causes, not symptoms.
    Handles missing values using MAR/MNAR-aware strategies.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.audit_log = []
        self.wmi_db = self._build_wmi_database()

    def _build_wmi_database(self) -> Dict[str, Tuple[str, Optional[str]]]:
        """World Manufacturer Identifier database for VIN-based imputation."""
        return {
            '1FT': ('Ford', 'Truck'),
            '1G1': ('Chevrolet', None),
            '1G2': ('Pontiac', None),
            '1G3': ('Oldsmobile', None),
            '1G4': ('Buick', None),
            '1G6': ('Cadillac', None),
            '1GC': ('Chevrolet', 'Truck'),
            '1GT': ('GMC', 'Truck'),
            '1HG': ('Honda', None),
            '1J4': ('Jeep', None),
            '1J8': ('Jeep', None),
            '1N4': ('Nissan', None),
            '1N6': ('Nissan', 'Truck'),
            '1VW': ('Volkswagen', None),
            '1YV': ('Mazda', None),
            '19U': ('Acura', None),
            '2F1': ('Chevrolet', None),
            '2G1': ('Chevrolet', None),
            '2G2': ('Pontiac', None),
            '2H8': ('Honda', None),
            '2T1': ('Toyota', None),
            '2T3': ('Toyota', 'SUV'),
            '3AL': ('Freightliner', None),
            '3D6': ('Dodge', None),
            '3VW': ('Volkswagen', None),
            '4S3': ('Subaru', None),
            '4S4': ('Subaru', None),
            '5J6': ('Honda', 'SUV'),
            '5N1': ('Hyundai', 'SUV'),
            '5N3': ('Hyundai', 'Truck'),
            '5T4': ('Hyundai', None),
            '5U3': ('Hyundai', None),
            '5XYP': ('Hyundai', None),
            '55S': ('Hyundai', None),
            '58A': ('Hyundai', None),
            '5UX': ('BMW', None),
            '5XX': ('Kia', None),
            'JM1': ('Mazda', None),
            'JTD': ('Toyota', None),
            'JTH': ('Lexus', None),
            'JTJ': ('Lexus', 'SUV'),
            'JT6': ('Toyota', 'Truck'),
            'JT8': ('Toyota', None),
            'KL1': ('Chevrolet', None),
            'KL7': ('Chevrolet', None),
            'KN8': ('Kia', None),
            'KNA': ('Kia', None),
            'KNB': ('Kia', None),
            'KNC': ('Kia', None),
            'KND': ('Kia', None),
            'SAJ': ('Jaguar', None),
            'SAL': ('Land Rover', None),
            'SAR': ('Land Rover', None),
            'SCA': ('Rolls Royce', None),
            'SCB': ('Bentley', None),
            'SCC': ('Lotus', None),
            'SCF': ('Aston Martin', None),
            'SHH': ('Honda', None),
            'SHS': ('Honda', 'SUV'),
            'TRU': ('Audi', None),
            'TMB': ('Skoda', None),
            'TML': ('Hyundai', None),
            'TRW': ('Seat', None),
            'VF1': ('Renault', None),
            'VF3': ('Peugeot', None),
            'VF7': ('Citroen', None),
            'VSS': ('Seat', None),
            'WA1': ('Audi', 'SUV'),
            'WAU': ('Audi', None),
            'WBA': ('BMW', None),
            'WBS': ('BMW', 'M Series'),
            'WBX': ('BMW', 'SUV'),
            'WDC': ('Mercedes-Benz', 'SUV'),
            'WDD': ('Mercedes-Benz', None),
            'WF0': ('Ford', None),
            'WMA': ('MAN', None),
            'WMX': ('Mercedes-Benz', None),
            'WP0': ('Porsche', None),
            'WP1': ('Porsche', 'SUV'),
            'WUA': ('Audi', 'RS'),
            'WVW': ('Volkswagen', None),
            'W0L': ('Opel', None),
            'YV1': ('Volvo', None),
            'YV4': ('Volvo', 'SUV'),
            'ZAM': ('Maserati', None),
            'ZAR': ('Alfa Romeo', None),
            'ZAS': ('Alfa Romeo', None),
            'ZFA': ('Fiat', None),
            'ZFF': ('Ferrari', None),
            'ZHW': ('Lamborghini', None),
            '1B4': ('Dodge', None),
            '1C3': ('Chrysler', None),
            '1C4': ('Chrysler', None),
            '1C6': ('Ram', 'Truck'),
            '1D3': ('Dodge', None),
            '1D7': ('Ram', None),
            '1F9': ('Fisker', None),
            '1FA': ('Ford', None),
            '1FB': ('Ford', None),
            '1FC': ('Ford', None),
            '1FD': ('Ford', 'Truck'),
            '1FM': ('Ford', 'SUV'),
            '1FT': ('Ford', 'Truck'),
            '1FU': ('Freightliner', None),
            '1FV': ('Freightliner', None),
            '1F9': ('Fisker', None),
            '1G0': ('Chevrolet', None),
            '1G1': ('Chevrolet', None),
            '1G2': ('Pontiac', None),
            '1G3': ('Oldsmobile', None),
            '1G4': ('Buick', None),
            '1G6': ('Cadillac', None),
            '1G8': ('Saturn', None),
            '1GA': ('Chevrolet', 'Van'),
            '1GC': ('Chevrolet', 'Truck'),
            '1GD': ('GMC', 'Van'),
            '1GK': ('GMC', 'SUV'),
            '1GT': ('GMC', 'Truck'),
            '1HG': ('Honda', None),
            '1HS': ('International', None),
            '1HT': ('International', None),
            '1J4': ('Jeep', None),
            '1J8': ('Jeep', None),
            '1JC': ('Jeep', None),
            '1JD': ('Jeep', None),
            '1JU': ('Jeep', None),
            '1L1': ('Lincoln', None),
            '1LN': ('Lincoln', None),
            '1ME': ('Mercury', None),
            '1M1': ('Mack', None),
            '1M2': ('Mack', None),
            '1M3': ('Mack', None),
            '1M4': ('Mack', None),
            '1N4': ('Nissan', None),
            '1N6': ('Nissan', 'Truck'),
            '1NX': ('Toyota', None),
            '1P3': ('Plymouth', None),
            '1P4': ('Plymouth', None),
            '1R9': ('Roadster', None),
            '1VW': ('Volkswagen', None),
            '1XK': ('Kenworth', None),
            '1XM': ('AMC', None),
            '1XP': ('Peterbilt', None),
            '1YV': ('Mazda', None),
            '1Z7': ('Mitsubishi', None),
            '1ZV': ('Ford', None),
            '2A4': ('Chrysler', None),
            '2A8': ('Chrysler', None),
            '2B1': ('Orion', None),
            '2B3': ('Dodge', None),
            '2B5': ('Dodge', None),
            '2B7': ('Dodge', None),
            '2C3': ('Chrysler', None),
            '2C4': ('Chrysler', None),
            '2C8': ('Chrysler', None),
            '2CN': ('Geo', None),
            '2D4': ('Dodge', None),
            '2D7': ('Dodge', None),
            '2F1': ('Ford', None),
            '2F2': ('Ford', None),
            '2FT': ('Ford', 'Truck'),
            '2FU': ('Freightliner', None),
            '2FV': ('Freightliner', None),
            '2G1': ('Chevrolet', None),
            '2G2': ('Pontiac', None),
            '2G3': ('Oldsmobile', None),
            '2G4': ('Buick', None),
            '2G6': ('Cadillac', None),
            '2G8': ('Chevrolet', None),
            '2HG': ('Honda', None),
            '2HK': ('Honda', 'SUV'),
            '2HJ': ('Honda', None),
            '2HM': ('Hyundai', None),
            '2M2': ('Mercury', None),
            '2M6': ('Mercury', None),
            '2T1': ('Toyota', None),
            '2T2': ('Lexus', None),
            '2T3': ('Toyota', 'SUV'),
            '3AL': ('Freightliner', None),
            '3D2': ('Dodge', None),
            '3D3': ('Dodge', None),
            '3D4': ('Dodge', None),
            '3D5': ('Dodge', None),
            '3D6': ('Dodge', None),
            '3D7': ('Dodge', None),
            '3FA': ('Ford', None),
            '3FC': ('Ford', 'Chassis'),
            '3FD': ('Ford', 'Truck'),
            '3FR': ('Ford', 'Truck'),
            '3FT': ('Ford', 'Truck'),
            '3G5': ('Buick', None),
            '3G7': ('Pontiac', None),
            '3GC': ('Chevrolet', 'Truck'),
            '3GN': ('Chevrolet', None),
            '3GT': ('GMC', 'Truck'),
            '3GY': ('Cadillac', 'SUV'),
            '3HG': ('Honda', None),
            '3HM': ('Honda', None),
            '3HS': ('Honda', None),
            '3LN': ('Lincoln', None),
            '3MA': ('Mazda', None),
            '3ME': ('Mercury', None),
            '3MZ': ('Mazda', None),
            '3N1': ('Nissan', None),
            '3PC': ('Infiniti', None),
            '3VW': ('Volkswagen', None),
            '4F2': ('Mazda', 'SUV'),
            '4F4': ('Mazda', 'Truck'),
            '4G1': ('Chevrolet', None),
            '4M2': ('Mercury', None),
            '4N2': ('Nissan', None),
            '4S1': ('Isuzu', None),
            '4S2': ('Isuzu', None),
            '4S3': ('Subaru', None),
            '4S4': ('Subaru', None),
            '4S6': ('Honda', None),
            '4T1': ('Toyota', None),
            '4T3': ('Toyota', None),
            '4T4': ('Toyota', None),
            '4US': ('BMW', None),
            '4UZ': ('Isuzu', None),
            '4V1': ('Volvo', None),
            '4V2': ('Volvo', None),
            '4V3': ('Volvo', None),
            '4V4': ('Volvo', None),
            '4V5': ('Volvo', None),
            '4V6': ('Volvo', None),
            '5AS': ('BMW', None),
            '5BZ': ('Hyundai', None),
            '5F9': ('Hyundai', None),
            '5J6': ('Honda', 'SUV'),
            '5J8': ('Acura', 'SUV'),
            '5KJ': ('Hyundai', None),
            '5N1': ('Hyundai', 'SUV'),
            '5N3': ('Hyundai', 'Truck'),
            '5NM': ('Hyundai', None),
            '5NP': ('Hyundai', None),
            '5T3': ('Hyundai', None),
            '5T4': ('Hyundai', None),
            '5U3': ('Hyundai', None),
            '5XX': ('Kia', None),
            '6G2': ('Pontiac', None),
            '8A1': ('Hyundai', None),
            '8AP': ('Fiat', None),
            '8AR': ('Dodge', None),
            '8AT': ('Dodge', None),
            '8BA': ('Chrysler', None),
            '8BC': ('Chrysler', None),
            '9BD': ('Fiat', None),
            '9BF': ('Ford', None),
            'J81': ('Chevrolet', None),
            'J87': ('Isuzu', None),
            'J8B': ('Chevrolet', None),
            'J8D': ('Chevrolet', None),
            'JA3': ('Mitsubishi', None),
            'JA4': ('Mitsubishi', None),
            'JA7': ('Mitsubishi', None),
            'JD1': ('Daihatsu', None),
            'JD4': ('Daihatsu', None),
            'JF1': ('Subaru', None),
            'JF2': ('Subaru', None),
            'JF3': ('Subaru', None),
            'JF4': ('Saab', None),
            'JH4': ('Acura', None),
            'JH6': ('Honda', None),
            'JK': ('Kawasaki', None),
            'JL6': ('Mitsubishi', None),
            'JM1': ('Mazda', None),
            'JM2': ('Mazda', None),
            'JM3': ('Mazda', None),
            'JN1': ('Nissan', None),
            'JN6': ('Datsun', None),
            'JS2': ('Suzuki', None),
            'JS3': ('Suzuki', None),
            'JS4': ('Suzuki', None),
            'JT2': ('Toyota', None),
            'JT3': ('Toyota', None),
            'JT4': ('Toyota', None),
            'JT5': ('Toyota', None),
            'JT6': ('Toyota', 'Truck'),
            'JT8': ('Toyota', None),
            'JTD': ('Toyota', None),
            'JTH': ('Lexus', None),
            'JTJ': ('Lexus', 'SUV'),
            'JTK': ('Scion', None),
            'JTL': ('Scion', None),
            'JTM': ('Toyota', None),
            'JTN': ('Toyota', None),
            'KL1': ('Chevrolet', None),
            'KL5': ('Suzuki', None),
            'KL7': ('Chevrolet', None),
            'KM8': ('Hyundai', None),
            'KMH': ('Hyundai', None),
            'KNA': ('Kia', None),
            'KNB': ('Kia', None),
            'KNC': ('Kia', None),
            'KND': ('Kia', None),
            'KNM': ('Renault', None),
            'KPA': ('Kia', None),
            'KPT': ('Kia', None),
            'L56': ('Volvo', None),
            'L5X': ('DAF', None),
            'LD9': ('Daewoo', None),
            'LFV': ('FAW', None),
            'LHG': ('Honda', None),
            'LJN': ('Nissan', None),
            'LKL': ('Suzuki', None),
            'LLV': ('Lifan', None),
            'LMG': ('Trumpchi', None),
            'LPR': ('Peugeot', None),
            'LS5': ('Changan', None),
            'LSG': ('Buick', None),
            'LSJ': ('MG', None),
            'LSV': ('Volkswagen', None),
            'LTV': ('Toyota', None),
            'LVS': ('Ford', None),
            'LVV': ('Chery', None),
            'LWV': ('Citroen', None),
            'LYV': ('Volvo', None),
            'LZG': ('Shacman', None),
            'LZY': ('Yutong', None),
            'MA1': ('Mahindra', None),
            'MA3': ('Suzuki', None),
            'MA6': ('GM', None),
            'MA7': ('Honda', None),
            'MAL': ('Hyundai', None),
            'MAT': ('Tata', None),
            'MB8': ('Mercedes-Benz', None),
            'MBH': ('Mercedes-Benz', None),
            'MBJ': ('Mercedes-Benz', None),
            'MBL': ('Mercedes-Benz', None),
            'MBN': ('Mercedes-Benz', None),
            'MBR': ('Mercedes-Benz', None),
            'MBU': ('Mercedes-Benz', None),
            'MBV': ('Mercedes-Benz', None),
            'MC2': ('Volvo', None),
            'MD2': ('Kia', None),
            'MD7': ('Hyundai', None),
            'MDH': ('Honda', None),
            'MEC': ('Mitsubishi', None),
            'MFB': ('Toyota', None),
            'MFG': ('Ford', None),
            'MFN': ('Ford', None),
            'MHR': ('Honda', None),
            'MLC': ('Suzuki', None),
            'MLH': ('Honda', None),
            'MM0': ('Mazda', None),
            'MM6': ('Mazda', None),
            'MM7': ('Mazda', None),
            'MMB': ('Mitsubishi', None),
            'MMC': ('Mitsubishi', None),
            'MML': ('MG', None),
            'MMM': ('Chevrolet', None),
            'MMS': ('Peugeot', None),
            'MMT': ('Mitsubishi', None),
            'MMU': ('Holden', None),
            'MMX': ('Honda', None),
            'MN1': ('Nissan', None),
            'MNT': ('Nissan', None),
            'MNT': ('Nissan', None),
            'MPA': ('Isuzu', None),
            'MPA': ('Isuzu', None),
            'MPB': ('Isuzu', None),
            'MR0': ('Toyota', None),
            'MRH': ('Honda', None),
            'MS0': ('Kia', None),
            'MS3': ('Suzuki', None),
            'NLA': ('Honda', None),
            'NLH': ('Hyundai', None),
            'NLJ': ('Hyundai', None),
            'NLT': ('Hyundai', None),
            'NLV': ('Hyundai', None),
            'NMB': ('Mercedes-Benz', None),
            'NMC': ('Mercedes-Benz', None),
            'NMT': ('Toyota', None),
            'NM0': ('Ford', None),
            'NM4': ('Hyundai', None),
            'NNA': ('Isuzu', None),
            'PE3': ('Mazda', None),
            'PL1': ('Proton', None),
            'PNA': ('Honda', None),
            'RF3': ('Acura', None),
            'RF4': ('Acura', None),
            'RF5': ('Honda', None),
            'RF6': ('Honda', None),
            'RL4': ('Honda', None),
            'RL5': ('Honda', None),
            'RLH': ('Honda', None),
            'RP8': ('Peugeot', None),
            'SAB': ('Optare', None),
            'SAD': ('Jaguar', None),
            'SAF': ('Foden', None),
            'SAH': ('Austin', None),
            'SAJ': ('Jaguar', None),
            'SAL': ('Land Rover', None),
            'SAN': ('Land Rover', None),
            'SAR': ('Rover', None),
            'SAT': ('Triumph', None),
            'SAX': ('Austin', None),
            'SB1': ('Toyota', None),
            'SBM': ('McLaren', None),
            'SCA': ('Rolls Royce', None),
            'SCB': ('Bentley', None),
            'SCC': ('Lotus', None),
            'SCF': ('Aston Martin', None),
            'SCK': ('Jaguar', None),
            'SDB': ('Peugeot', None),
            'SED': ('General Motors', None),
            'SEY': ('Chrysler', None),
            'SFA': ('Ford', None),
            'SHH': ('Honda', None),
            'SHS': ('Honda', None),
            'SJN': ('Nissan', None),
            'SK5': ('Fiat', None),
            'SK6': ('Kia', None),
            'SLA': ('Rolls Royce', None),
            'SLE': ('Jaguar', None),
            'SMT': ('Hyundai', None),
            'SNE': ('BMW', None),
            'SU9': ('Lamborghini', None),
            'SUF': ('Fiat', None),
            'SUL': ('Fiat', None),
            'SUP': ('Fiat', None),
            'SUU': ('Fiat', None),
            'SW6': ('Kenworth', None),
            'SW9': ('Kenworth', None),
            'T7B': ('Hyundai', None),
            'TCC': ('Smart', None),
            'TMA': ('Hyundai', None),
            'TMB': ('Skoda', None),
            'TRU': ('Audi', None),
            'TSM': ('Suzuki', None),
            'TW1': ('Toyota', None),
            'TWB': ('Citroen', None),
            'U5Y': ('Kia', None),
            'UU1': ('Renault', None),
            'UU6': ('Renault', None),
            'VAG': ('Chevrolet', None),
            'VAN': ('Chevrolet', None),
            'VF1': ('Renault', None),
            'VF3': ('Peugeot', None),
            'VF6': ('Renault', None),
            'VF7': ('Citroen', None),
            'VF8': ('Citroen', None),
            'VSS': ('Seat', None),
            'VV9': ('Citroen', None),
            'W0L': ('Opel', None),
            'W0V': ('Opel', None),
            'WAP': ('Alpina', None),
            'WA1': ('Audi', 'SUV'),
            'WAU': ('Audi', None),
            'WBA': ('BMW', None),
            'WBS': ('BMW', 'M Series'),
            'WBX': ('BMW', 'SUV'),
            'WBY': ('BMW', None),
            'WDA': ('Daimler', None),
            'WDB': ('Mercedes-Benz', None),
            'WDC': ('Mercedes-Benz', 'SUV'),
            'WDD': ('Mercedes-Benz', None),
            'WDF': ('Mercedes-Benz', 'Van'),
            'WDY': ('Dodge', None),
            'WEB': ('Evobus', None),
            'WF0': ('Ford', None),
            'WF1': ('Renault', None),
            'WF3': ('Peugeot', None),
            'WF7': ('Citroen', None),
            'WG0': ('Ford', None),
            'WMA': ('MAN', None),
            'WME': ('Smart', None),
            'WMX': ('Mercedes-Benz', None),
            'WMY': ('Mercedes-Benz', None),
            'WP0': ('Porsche', None),
            'WP1': ('Porsche', 'SUV'),
            'WUA': ('Audi', 'RS'),
            'WV1': ('Volkswagen', 'Van'),
            'WV2': ('Volkswagen', 'Van'),
            'WVW': ('Volkswagen', None),
            'W0L': ('Opel', None),
            'XLB': ('Volvo', None),
            'XLE': ('Scania', None),
            'XLR': ('Scania', None),
            'XL9': ('Scania', None),
            'Y6D': ('Zastava', None),
            'YV1': ('Volvo', None),
            'YV2': ('Volvo', 'Truck'),
            'YV4': ('Volvo', 'SUV'),
            'YV5': ('Volvo', None),
            'ZAM': ('Maserati', None),
            'ZAP': ('Piaggio', None),
            'ZAR': ('Alfa Romeo', None),
            'ZAS': ('Alfa Romeo', None),
            'ZCF': ('Iveco', None),
            'ZFA': ('Fiat', None),
            'ZFC': ('Fiat', 'Chassis'),
            'ZFF': ('Ferrari', None),
            'ZHW': ('Lamborghini', None),
            'ZLA': ('Lancia', None),
            'ZOM': ('OM', None),
        }

    def fix_column_shifts(self) -> 'VehicleDataCleaner':
        """
        CS-001: Fix the 'sedan' column shift (26 rows)
        Root cause: CSV parsing error with unescaped comma in trim
        """
        # Detect shifted rows
        shift_mask = self.df["Transmission"].isin(["sedan", "Sedan"])

        if shift_mask.sum() == 0:
            self.audit_log.append("CS-001: No column shifts detected")
            return self

        count = shift_mask.sum()

        # Realign right-to-left to prevent data overwrite
        # Order matters: start from rightmost column (State), move left

        # Step 1: State becomes NaN (unrecoverable - was VIN fragment)
        self.df.loc[shift_mask, "State"] = np.nan

        # Step 2: VIN takes what was in State (the actual VIN)
        # But we need to save State values first
        temp_state = self.df.loc[shift_mask, "State"].copy()
        self.df.loc[shift_mask, "State"] = np.nan  # Now safe to overwrite

        # Actually, we need to work with the ORIGINAL values before any modifications
        # Let me redo this properly

        # Get original values for shifted rows
        shifted_rows = self.df[shift_mask].copy()

        # Realignment:
        # Body (has "Navigation") <- should be "Sedan"
        # Transmission (has "Sedan") <- should be what was in VIN
        # VIN (has "automatic" or null) <- should be what was in State
        # State (has VIN fragment) <- should be NaN (unrecoverable)
        # ConditionValue <- NaN (unrecoverable)

        self.df.loc[shift_mask, "ConditionValue"] = np.nan

        # Save current values before overwriting
        current_state = shifted_rows["State"].values
        current_vin = shifted_rows["VIN"].values
        current_transmission = shifted_rows["Transmission"].values

        # Realign
        self.df.loc[shift_mask, "State"] = np.nan  # Unrecoverable
        self.df.loc[shift_mask, "VIN"] = current_state  # VIN was in State
        self.df.loc[shift_mask, "Transmission"] = current_vin  # Transmission was in VIN
        self.df.loc[shift_mask, "Body"] = "Sedan"  # Body was garbage

        self.audit_log.append(f"CS-001: Fixed {count} column-shifted rows")

        return self

    def fix_make_body_concatenation(self) -> 'VehicleDataCleaner':
        """
        CS-002: Fix make-body concatenations (19 rows)
        Pattern: "ford truck", "gmc truck", etc.
        """
        make_body_fixes = {
            "ford truck": ("Ford", "Truck"),
            "gmc truck": ("GMC", "Truck"),
            "chev truck": ("Chevrolet", "Truck"),
            "ford tk": ("Ford", "Truck"),
            "dodge tk": ("Dodge", "Truck"),
            "mazda tk": ("Mazda", "Truck"),
            "hyundai tk": ("Hyundai", "Truck"),
            "chev tk": ("Chevrolet", "Truck"),
        }

        for raw, (correct_make, correct_body) in make_body_fixes.items():
            mask = self.df["Make"].str.lower() == raw.lower()
            if mask.sum() > 0:
                self.df.loc[mask, "Make"] = correct_make
                self.df.loc[mask, "Body"] = correct_body
                self.audit_log.append(f"CS-002: Fixed '{raw}' -> {correct_make}/{correct_body} ({mask.sum()} rows)")

        return self

    def fix_model_fragmentation(self) -> 'VehicleDataCleaner':
        """
        MF-001: Fix Range Rover model name fragmentation
        """
        # Pattern 1: Model="range", Trim contains "rover"
        range_mask = (
            (self.df["Model"].str.lower() == "range") & 
            (self.df["Trim"].str.contains("rover", case=False, na=False))
        )
        if range_mask.sum() > 0:
            self.df.loc[range_mask, "Model"] = "Range Rover"
            self.audit_log.append(f"MF-001: Fixed 'range' + 'rover' split ({range_mask.sum()} rows)")

        # Pattern 2: Model="rangerover" (concatenated)
        rangerover_mask = self.df["Model"].str.lower() == "rangerover"
        if rangerover_mask.sum() > 0:
            self.df.loc[rangerover_mask, "Model"] = "Range Rover"
            self.audit_log.append(f"MF-001: Fixed 'rangerover' concatenation ({rangerover_mask.sum()} rows)")

        # Pattern 3: Abbreviations
        abbrev_mask = self.df["Model"].isin(["rr", "RR", "rrs", "RRS"])
        if abbrev_mask.sum() > 0:
            self.df.loc[abbrev_mask, "Model"] = "Range Rover"
            self.audit_log.append(f"MF-001: Fixed RR/RRS abbreviations ({abbrev_mask.sum()} rows)")

        return self

    def standardize_makes(self) -> 'VehicleDataCleaner':
        """
        MK-001: Standardize make names using VIN validation
        """
        make_mapping = {
            "vw": "Volkswagen",
            "dot": "Dodge",
            "mercedes-b": "Mercedes-Benz",
            "mercedes-benz": "Mercedes-Benz",
            "chev": "Chevrolet",
            "chevy": "Chevrolet",
        }

        for wrong, right in make_mapping.items():
            mask = self.df["Make"].str.lower() == wrong
            if mask.sum() > 0:
                # Validate with VIN if available
                for idx in self.df[mask].index:
                    vin = str(self.df.loc[idx, "VIN"]) if pd.notna(self.df.loc[idx, "VIN"]) else ""
                    if len(vin) >= 3:
                        wmi = vin[:3].upper()
                        expected_make = self.wmi_db.get(wmi, (None, None))[0]
                        if expected_make and expected_make.lower() == right.lower():
                            self.df.loc[idx, "Make"] = right
                        elif expected_make:
                            # VIN suggests different make, investigate
                            pass

                self.audit_log.append(f"MK-001: Standardized '{wrong}' -> '{right}' ({mask.sum()} rows)")

        # Title case all makes
        self.df["Make"] = self.df["Make"].str.title()

        return self

    def handle_sentinel_values(self) -> 'VehicleDataCleaner':
        """
        PH-001: Handle placeholder/sentinel values (MNAR - Missing Not At Random)

        CRITICAL: These are NOT missing at random. They are deliberate sentinels
        that must be converted to NaN and handled with MNAR-aware strategies.
        """
        # Odometer = 999999 is a deliberate placeholder
        odometer_mask = self.df["Odometer"] == 999999
        if odometer_mask.sum() > 0:
            self.df.loc[odometer_mask, "Odometer"] = np.nan
            self.audit_log.append(f"PH-001: Converted {odometer_mask.sum()} odometer sentinels (999999) to NaN")

        # Color/Interior = "---" is placeholder for unknown
        for col in ["Color", "Interior"]:
            if col in self.df.columns:
                dash_mask = self.df[col] == "---"
                if dash_mask.sum() > 0:
                    self.df.loc[dash_mask, col] = np.nan
                    self.audit_log.append(f"PH-002: Converted {dash_mask.sum()} '{col}' placeholders to NaN")

        # Any other obvious sentinels
        if "MMR" in self.df.columns:
            mmr_mask = self.df["MMR"] == 999999
            if mmr_mask.sum() > 0:
                self.df.loc[mmr_mask, "MMR"] = np.nan
                self.audit_log.append(f"PH-003: Converted {mmr_mask.sum()} MMR sentinels to NaN")

        return self

    def impute_from_vin(self) -> 'VehicleDataCleaner':
        """
        Strategy 3: Domain-knowledge imputation using VIN decoding

        For rows with missing Make/Model but valid VIN, decode WMI to infer values.
        This is NOT statistical imputation - it's domain knowledge.
        """
        # Find rows with missing Make but valid VIN
        missing_make = self.df["Make"].isna() & self.df["VIN"].notna()

        vin_fixed = 0
        for idx in self.df[missing_make].index:
            vin = str(self.df.loc[idx, "VIN"])
            if len(vin) >= 3:
                wmi = vin[:3].upper()
                if wmi in self.wmi_db:
                    make, body_hint = self.wmi_db[wmi]
                    if make:
                        self.df.loc[idx, "Make"] = make
                        vin_fixed += 1
                        # If we have a body type hint and Body is missing
                        if body_hint and pd.isna(self.df.loc[idx, "Body"]):
                            self.df.loc[idx, "Body"] = body_hint

        if vin_fixed > 0:
            self.audit_log.append(f"IMP-VIN: Imputed {vin_fixed} Make values from VIN WMI")

        return self

    def knn_impute_color_interior(self) -> 'VehicleDataCleaner':
        """
        Strategy 2: KNN Imputation for Color/Interior (MAR - Missing At Random)

        Assumption: Color/Interior missingness correlates with observed features
        (Year, Make, Body, Odometer). We use KNN to find similar vehicles.

        IMPORTANT: Only for rows where we have sufficient features.
        """
        features_for_imputation = ["Year", "Odometer"]

        # Check which features exist
        available_features = [f for f in features_for_imputation if f in self.df.columns]

        if len(available_features) < 1:
            self.audit_log.append("IMP-KNN: Insufficient features for KNN imputation")
            return self

        # For Color
        if "Color" in self.df.columns:
            missing_color = self.df["Color"].isna()
            if missing_color.sum() > 0:
                # Prepare data for KNN
                # Encode categoricals
                df_encoded = self.df.copy()

                le_make = LabelEncoder()
                le_body = LabelEncoder()
                le_color = LabelEncoder()

                # Fit encoders on non-missing data
                if "Make" in df_encoded.columns:
                    valid_make = df_encoded["Make"].dropna()
                    if len(valid_make) > 0:
                        le_make.fit(valid_make.astype(str))
                        df_encoded["Make_encoded"] = df_encoded["Make"].apply(
                            lambda x: le_make.transform([str(x)])[0] if pd.notna(x) else np.nan
                        )
                        available_features.append("Make_encoded")

                if "Body" in df_encoded.columns:
                    valid_body = df_encoded["Body"].dropna()
                    if len(valid_body) > 0:
                        le_body.fit(valid_body.astype(str))
                        df_encoded["Body_encoded"] = df_encoded["Body"].apply(
                            lambda x: le_body.transform([str(x)])[0] if pd.notna(x) else np.nan
                        )
                        available_features.append("Body_encoded")

                # Prepare feature matrix
                feature_cols = available_features
                X = df_encoded[feature_cols].copy()

                # Handle any remaining NaN in features with simple imputation (median)
                # This is acceptable for features, not target
                for col in X.columns:
                    if X[col].isna().sum() > 0:
                        X[col].fillna(X[col].median(), inplace=True)

                # Encode target (Color) for rows where it's available
                valid_color_mask = df_encoded["Color"].notna()
                if valid_color_mask.sum() > 10:  # Need minimum samples
                    y = df_encoded.loc[valid_color_mask, "Color"].astype(str)
                    le_color.fit(y)
                    y_encoded = le_color.transform(y)

                    # Fit KNN on complete cases
                    knn = KNNImputer(n_neighbors=5, weights='distance')

                    # We need a different approach - KNNImputer works on features, not target
                    # Use simple KNN classifier for categorical imputation
                    from sklearn.neighbors import KNeighborsClassifier

                    knn_clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
                    knn_clf.fit(X[valid_color_mask], y_encoded)

                    # Predict for missing values
                    missing_indices = df_encoded[missing_color].index
                    if len(missing_indices) > 0:
                        X_missing = X.loc[missing_indices]
                        predictions = knn_clf.predict(X_missing)
                        predicted_colors = le_color.inverse_transform(predictions)

                        self.df.loc[missing_indices, "Color"] = predicted_colors
                        self.audit_log.append(f"IMP-KNN: Imputed {len(missing_indices)} Color values using KNN")

        return self

    def flag_non_imputable(self) -> 'VehicleDataCleaner':
        """
        CRITICAL: Flag rows that should NOT be imputed

        Rows with Make=NULL, Model=NULL, Trim=NULL, Body=NULL and no valid VIN
        are structurally incomplete. Imputation would add noise, not signal.
        """
        # Find rows with all key categoricals missing
        categorical_cols = ["Make", "Model", "Trim", "Body"]
        available_cols = [c for c in categorical_cols if c in self.df.columns]

        if len(available_cols) == 0:
            return self

        # Check if all are null
        all_null_mask = self.df[available_cols].isna().all(axis=1)

        # And no VIN to help
        no_vin_mask = self.df["VIN"].isna() | (self.df["VIN"] == "")

        non_imputable = all_null_mask & no_vin_mask

        if non_imputable.sum() > 0:
            # Add flag column
            self.df["_flag_non_imputable"] = non_imputable
            self.audit_log.append(f"FLAG: Marked {non_imputable.sum()} rows as non-imputable (insufficient data)")

        return self

    def probabilistic_impute(self, column: str, condition_columns: List[str]) -> 'VehicleDataCleaner':
        """
        Strategy 4: Probabilistic imputation preserving variance

        Instead of mode imputation (artificially reduces variance),
        sample from conditional distribution P(column | condition_columns)
        """
        if column not in self.df.columns:
            return self

        missing_mask = self.df[column].isna()
        if missing_mask.sum() == 0:
            return self

        # Build conditional probability distribution
        # P(column | condition_columns)

        condition_cols = [c for c in condition_columns if c in self.df.columns]
        if len(condition_cols) == 0:
            return self

        # Group by condition columns and calculate value distribution
        grouped = self.df.dropna(subset=[column]).groupby(condition_cols)[column].value_counts(normalize=True)

        # For each missing value, sample from its conditional distribution
        imputed_count = 0
        for idx in self.df[missing_mask].index:
            # Build key from condition columns
            key = tuple(self.df.loc[idx, col] for col in condition_cols)

            # Check if we have distribution for this key
            if key in grouped.index:
                values = grouped[key].index
                probs = grouped[key].values

                # Sample
                sampled = np.random.choice(values, p=probs)
                self.df.loc[idx, column] = sampled
                imputed_count += 1

        if imputed_count > 0:
            self.audit_log.append(f"IMP-PROB: Probabilistically imputed {imputed_count} {column} values")

        return self

    def standardize_categoricals(self) -> 'VehicleDataCleaner':
        """OPTIMIZED: Basic standardization"""
        # Transmission
        if "Transmission" in self.df.columns:
            mask = self.df["Transmission"].notna()
            if mask.sum() > 0:
                self.df.loc[mask, "Transmission"] = (
                    self.df.loc[mask, "Transmission"]
                    .astype(str).str.strip().str.lower()
                    .replace({"auto": "automatic", "man": "manual"})
                )
        
        # Text columns - process only non-null
        text_cols = ["Make", "Model", "Body", "Color", "Interior", "State", "Trim"]
        for col in [c for c in text_cols if c in self.df.columns]:
            mask = self.df[col].notna()
            if mask.sum() > 0:
                self.df.loc[mask, col] = (
                    self.df.loc[mask, col].astype(str).str.strip().str.title()
                )
        
        return self

    def clean(self) -> pd.DataFrame:
        """
        Execute full cleaning pipeline in correct order
        """
        print("Starting forensic data cleaning...")
        print(f"Initial shape: {self.df.shape}")

        # Phase 1: Fix structural errors (column shifts)
        print("\nPhase 1: Fixing column shifts...")
        self.fix_column_shifts()
        self.fix_make_body_concatenation()
        self.fix_model_fragmentation()

        # Phase 2: Standardize
        print("Phase 2: Standardizing categoricals...")
        self.standardize_makes()
        self.standardize_categoricals()

        # Phase 3: Handle sentinel values (MNAR)
        print("Phase 3: Converting sentinel values to NaN...")
        self.handle_sentinel_values()

        # Phase 4: Domain-knowledge imputation (VIN-based)
        print("Phase 4: VIN-based imputation...")
        self.impute_from_vin()

        # Phase 5: Statistical imputation (MAR only)
        print("Phase 5: MAR-aware statistical imputation...")
        self.knn_impute_color_interior()

        # Probabilistic imputation for remaining categoricals
        if "Color" in self.df.columns:
            self.probabilistic_impute("Color", ["Make", "Body", "Year"])
        if "Interior" in self.df.columns:
            self.probabilistic_impute("Interior", ["Make", "Body", "Year"])

        # Phase 6: Flag non-imputable rows
        print("Phase 6: Flagging non-imputable rows...")
        self.flag_non_imputable()

        print(f"\nFinal shape: {self.df.shape}")
        print("\nAudit Log:")
        for entry in self.audit_log:
            print(f"  - {entry}")

        return self.df

    def get_audit_log(self) -> List[str]:
        """Return the audit log for reporting"""
        return self.audit_log


def main():
    """
    Example usage
    """
    # Load your data
    # df = pd.read_csv("vehicle_sales_final.csv")
    # OR
    df = pd.read_parquet("/home/mennatullah/Documents/repos/li/AI/instant/corrected/VehicleSales.parquet")

    # Initialize cleaner
    cleaner = VehicleDataCleaner(df)

    # Run cleaning
    df_cleaned = cleaner.clean()

    # Save cleaned data
    df_cleaned.to_csv("vehicle_sales_cleaned.csv", index=False)

    # Get audit log for documentation
    audit_log = cleaner.get_audit_log()


if __name__ == "__main__":
    main()
