"""
Integrated Mining Claim Redistribution System

This script combines three workflows:
1. Filter and translate Midland claims (Casault filter)
2. Merge Wallbridge and Midland datasets
3. Run redistribution simulation with life expectancy analysis

Takes Midland and Wallbridge Excel files as input, produces merged CSV and redistribution analysis.

By default, only processes claims from these projects:
- CASAULT, MARTINIERE, FENELON, GRASSET, HARRI, DOIGT

Both the claims shapefile and property outlines shapefile are filtered to only show these projects.
"""

# Import libraries
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import pandas as pd
import seaborn as sns
import os
import sys
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Optional, List, Dict, Any, Tuple
import re
import copy
import argparse
import folium
from folium.plugins import Draw
import json
from tqdm import tqdm
import csv
import stat
from pathlib import Path
from io import BytesIO

# Try to import reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Spacer, Image as RLImage, Table, TableStyle, BaseDocTemplate, Frame, PageTemplate, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib import colors as rl_colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: ReportLab not available. PDF generation will be disabled.")
    print("Install with: pip install reportlab")

# Fix tqdm for frozen executables (disable progress bars when stdout is None)
if getattr(sys, 'frozen', False) or sys.stdout is None:
    # Running as compiled executable - disable tqdm progress bars
    tqdm = lambda iterable, *args, **kwargs: iterable

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    'MAX_DISTANCE': 3900.0,
    'MAX_YEAR': 2060,
    'MAX_RENEWALS': 6,
    'SCORING_MODE': 'earliest_expiry',
    'SCORING_WEIGHTS': {'surplus': 0.3, 'distance': 0.7},
    'INCLUDED_PROJECTS': ['CASAULT', 'MARTINIERE', 'FENELON', 'GRASSET', 'HARRI', 'DOIGT'],
    'MIDLAND_XLSX': r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Gestim_Midland_09012026.xlsx",
    'WALLBRIDGE_XLSX': r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Translated\Gestim_Wallbridge_090126.xlsx",
    'PROPERTY_CSV': r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Property_to_Claim.csv",
    'SHP_PATH': r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Shapefile\gsm_claims_20250703.shp",
    'OUTLINES_SHP': r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Shapefile\wmc_property_outlines.shp",
    'OUTPUT_DIR': r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Redistribution",
    'TEMP_DIR': r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Temp",
    'LOG_DIR': r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Logs",
    'CURRENT_DATE': '2026-01-09'
}

# Translation dictionaries
TRANSLATIONS = {
    'Feuillet': 'Map Sheet',
    'Nom canton/seigneurie': 'Canton/Seigneury Name',
    'Code canton/seigneurie': 'Canton/Seigneury Code',
    'Type de polygone': 'Polygon Type',
    'Rang/Bloc/Parcelle': 'Range/Block/Plot',
    'Rangée/Bloc': 'Row/Block',
    'Colonne/Lot': 'Column/Lot',
    'Partie': 'Part',
    'Superficie Polygone': 'Polygon Area',
    'Type de titre': 'Title Type',
    'No titre': 'Title Number',
    'Statut du titre': 'Title Status',
    'Date de jalonnement': 'Staking Date',
    'Date d\'inscription': 'Registration Date',
    'Date d\'expiration': 'Expiration Date',
    'Nombre d\'échéances': 'Number of Deadlines',
    'Nombre de renouvellements': 'Number of Renewals',
    'Superficie (Ha)': 'Area (Ha)',
    'Acte(s) relatif(s)': 'Related Act(s)',
    'Excédents': 'Surpluses',
    'Travaux requis': 'Required Works',
    'Droits requis': 'Required Rights',
    'Détenteur(s) (Nom, Numéro et Pourcentage)': 'Holder(s) (Name, Number, and Percentage)',
    'Feuillet site SMS': 'SMS Site Map Sheet',
    'No site SMS': 'SMS Site Number',
    'Renouvellement en traitement': 'Renewal in Progress',
    'Travaux en traitement': 'Works in Progress',
    'Transfert_Titre': 'Title Transfer',
    'Description': 'Description',
    'Commentaire localisation': 'Location Comment',
    'Commentaire contrainte': 'Constraint Comment',
    'Conversion/Substitution de droits exclusifs d\'exploration': 'Conversion/Substitution of Exclusive Exploration Rights',
    'Fusion de droits exclusifs d\'exploration': 'Merger of Exclusive Exploration Rights',
    'Découverte U3O8': 'U3O8 Discovery',
    'Territoire incompatible': 'Incompatible Territory',
    'Région administrative': 'Administrative Region',
    'MRC': 'MRC',
    'Municipalité': 'Municipality',
    'Dates et montants de péremption': 'Expiration Dates and Amounts'
}

DATA_TRANSLATIONS = {
    'Title Status': {'Actif': 'Active'},
    'Related Act(s)': {'Oui': 'Yes', 'Non': 'No'},
    'Renewal in Progress': {'Non': 'No'},
    'Works in Progress': {'Oui': 'Yes', 'Non': 'No'},
    'Title Transfer': {'Oui': 'Yes', 'Non': 'No'},
    'Conversion/Substitution of Exclusive Exploration Rights': {'Non': 'No'},
    'Merger of Exclusive Exploration Rights': {'Non': 'No'},
    'U3O8 Discovery': {'Non': 'No'},
    'Incompatible Territory': {'Non': 'No'},
    'Municipality': {
        "Gouvernement régional d'Eeyou Istchee Baie-James": 'Eeyou Istchee James Bay Regional Government'
    },
    'Holder(s) (Name, Number, and Percentage)': {
        'Wallbridge Mining Company Limited (100085) 100 % (représentant)': 'Wallbridge Mining Company Limited (100085) 100% (representative)'
    },
    'Description': {
        'Décision 32-22376 - Période de validité suspendue du 9 avril 2020 au 9 avril 2021, Déc. minis. 2020-04-09, (32-22283)': 'Decision 32-22376 - Validity period suspended from April 9, 2020, to April 9, 2021, Ministerial Decision 2020-04-09, (32-22283)'
    }
}

# ==============================================================================
# STYLE CONSTANTS FOR PDF REPORT
# ==============================================================================

class ModernStyle:
    """Modern color scheme and styling constants for PDF reports."""
    # Color palette - Modern professional theme
    PRIMARY = '#2C3E50'      # Dark blue-gray
    SECONDARY = '#3498DB'    # Bright blue
    SUCCESS = '#27AE60'      # Green
    WARNING = '#F39C12'      # Orange
    DANGER = '#E74C3C'       # Red
    INFO = '#16A085'         # Teal
    LIGHT = '#ECF0F1'        # Light gray
    DARK = '#34495E'         # Darker gray
    WHITE = '#FFFFFF'

# ==============================================================================
# CLAIM NUMBERS
# ==============================================================================

# Casault claim numbers for Midland filtering
CASAULT_CLAIM_NUMBERS = [
    '2208453', '2208454', '2208455', '2208456', '2208457', '2208458', '2208459', '2208460', '2208461', '2208462',
    '2208463', '2208464', '2208465', '2208466', '2208467', '2208468', '2208469', '2208470', '2208471', '2208472',
    '2208473', '2208474', '2208475', '2208476', '2208477', '2208478', '2208479', '2208480', '2208481', '2208482',
    '2208483', '2208484', '2208485', '2208486', '2208487', '2208488', '2208489', '2208490', '2208491', '2208492',
    '2208523', '2208524', '2208525', '2208526', '2208527', '2208528', '2208529', '2208530', '2208531', '2208532',
    '2208533', '2208534', '2208535', '2208536', '2208537', '2208538', '2208539', '2208540', '2208541', '2208542',
    '2208543', '2208544', '2208545', '2208546', '2208547', '2208548', '2208549', '2208550', '2208551', '2208552',
    '2208553', '2208554', '2208555', '2208556', '2208557', '2208558', '2208559', '2208560', '2208561', '2208562',
    '2208565', '2208566', '2208567', '2208568', '2208569', '2208570', '2208571', '2208572', '2211287', '2211288',
    '2211289', '2211290', '2211291', '2211292', '2211293', '2211294', '2211295', '2211296', '2211297', '2211298',
    '2211299', '2211300', '2211301', '2211302', '2211303', '2214200', '2214201', '2214202', '2214203', '2214204',
    '2241673', '2247245', '2247246', '2247247', '2247248', '2247249', '2247250', '2247251', '2247252', '2247253',
    '2247254', '2247255', '2247256', '2247257', '2247258', '2247259', '2247260', '2247261', '2247262', '2247263',
    '2247264', '2247265', '2247266', '2247267', '2247268', '2247269', '2247270', '2247271', '2247272', '2247273',
    '2247274', '2247275', '2247276', '2247277', '2247278', '2247279', '2247280', '2247281', '2247282', '2247283',
    '2247284', '2271264', '2271265', '2273155', '2273156', '2273157', '2273158', '2273159', '2273160', '2273161',
    '2273162', '2273163', '2273164', '2273165', '2273166', '2273167', '2276124', '2276125', '2276126', '2276127',
    '2276128', '2276129', '2276130', '2276131', '2276132', '2276133', '2276134', '2276135', '2276136', '2276137',
    '2276138', '2276139', '2276140', '2276141', '2276142', '2276143', '2276144', '2276145', '2276146', '2276147',
    '2276148', '2276149', '2276150', '2276151', '2276152', '2276153', '2276154', '2276155', '2276156', '2276157',
    '2276158', '2276159', '2276160', '2276161', '2282141', '2286321', '2286322', '2286323', '2286324', '2286325',
    '2286326', '2286327', '2286328', '2286329', '2286330', '2286331', '2286332', '2286777', '2286778', '2286779',
    '2286780', '2286781', '2286782', '2286783', '2286784', '2286785', '2286786', '2286787', '2286788', '2286790',
    '2286791', '2286792', '2286793', '2286794', '2286795', '2286796', '2286797', '2286798', '2286799', '2286800',
    '2286801', '2286802', '2286803', '2286804', '2294127', '2294128', '2313433', '2321964', '2322789', '2322790',
    '2322791', '2322792', '2322793', '2322794', '2322795', '2322796', '2322797', '2322798', '2322799', '2322800',
    '2322801', '2322802', '2322803', '2322804', '2322805', '2322806', '2322807', '2322808', '2322809', '2322810',
    '2322811', '2322812', '2322813', '2322814', '2322815', '2322816', '2322817', '2322818', '2322819', '2322820',
    '2322821', '2322822', '2322823', '2326101', '2326104', '2326106', '2384320', '2384321', '2384718', '2384719',
    '2384720', '2390766', '2395089', '2395090', '2395091', '2395092', '2395093', '2395094', '2436774', '2436775',
    '2437713', '2437714', '2437715', '2437720', '2438023', '2438024', '2439224', '2457675', '2457677', '2457678',
    '2457679', '2457680', '2513528', '2513529', '2517469', '2517470', '2539505', '2540266', '2540267', '2540268',
    '2540269', '2540270'
]

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    """Prompt user for yes/no input with default value."""
    default_str = "Y/n" if default else "y/N"
    while True:
        response = input(f"{prompt} [{default_str}]: ").strip().lower()
        if response == '':
            return default
        if response in ('y', 'yes'):
            return True
        if response in ('n', 'no'):
            return False
        print("Please enter 'y' or 'n'")


def validate_credits_excel(excel_path: str, valid_title_numbers: List[str]) -> Tuple[bool, List[str], Optional[pd.DataFrame]]:
    """
    Validate the credits Excel file for required columns, data types, and title existence.

    Expected columns:
    - Title Number (String): The claim title number
    - Amount (Float): The credit amount to allocate
    - Start Date (YYYY-MM-DD): The allocation start date

    Returns:
        Tuple of (is_valid, list_of_errors, dataframe_if_valid)
    """
    errors = []

    # Check file exists
    if not os.path.exists(excel_path):
        return False, [f"File not found: {excel_path}"], None

    # Try to read the Excel file
    try:
        df = pd.read_excel(excel_path, dtype=str, keep_default_na=False)
    except Exception as e:
        return False, [f"Failed to read Excel file: {str(e)}"], None

    # Check required columns exist (case-insensitive matching)
    required_columns = ['title number', 'amount', 'start date']
    column_mapping = {}

    df_columns_lower = {col.lower().strip(): col for col in df.columns}

    for req_col in required_columns:
        if req_col not in df_columns_lower:
            errors.append(f"Missing required column: '{req_col}' (found columns: {list(df.columns)})")
        else:
            column_mapping[req_col] = df_columns_lower[req_col]

    if errors:
        return False, errors, None

    # Rename columns to standard names
    df = df.rename(columns={
        column_mapping['title number']: 'Title Number',
        column_mapping['amount']: 'Amount',
        column_mapping['start date']: 'Start Date'
    })

    # Check for empty dataframe
    if len(df) == 0:
        return False, ["Excel file contains no data rows"], None

    # Validate Title Number column
    df['Title Number'] = df['Title Number'].astype(str).str.strip()
    empty_titles = df['Title Number'].isin(['', 'nan', 'None'])
    if empty_titles.any():
        errors.append(f"Empty title numbers found in rows: {list(df[empty_titles].index + 2)}")  # +2 for Excel row numbers (1-indexed + header)

    # Normalize title numbers for matching
    df['Title Number Normalized'] = df['Title Number'].apply(normalize_string)
    valid_titles_normalized = set(normalize_string(t) for t in valid_title_numbers)

    # Check if titles exist in the dataset
    invalid_titles = df[~df['Title Number Normalized'].isin(valid_titles_normalized)]
    if len(invalid_titles) > 0:
        invalid_list = invalid_titles['Title Number'].tolist()
        if len(invalid_list) > 10:
            errors.append(f"Title numbers not found in dataset ({len(invalid_list)} total): {invalid_list[:10]}... and {len(invalid_list)-10} more")
        else:
            errors.append(f"Title numbers not found in dataset: {invalid_list}")

    # Validate Amount column - must be numeric and positive
    amount_errors = []
    amounts = []
    for idx, val in enumerate(df['Amount']):
        try:
            # Handle comma as decimal separator
            val_clean = str(val).replace(',', '.').replace('$', '').replace(' ', '').strip()
            if val_clean in ['', 'nan', 'None']:
                amount_errors.append(f"Row {idx + 2}: Empty amount")
                amounts.append(None)
            else:
                amount = float(val_clean)
                if amount <= 0:
                    amount_errors.append(f"Row {idx + 2}: Amount must be positive (got {amount})")
                    amounts.append(None)
                else:
                    amounts.append(amount)
        except ValueError:
            amount_errors.append(f"Row {idx + 2}: Invalid amount format '{val}'")
            amounts.append(None)

    df['Amount Parsed'] = amounts

    if amount_errors:
        if len(amount_errors) > 5:
            errors.append(f"Amount validation errors ({len(amount_errors)} total): {amount_errors[:5]}... and more")
        else:
            errors.extend(amount_errors)

    # Validate Start Date column - must be valid date in YYYY-MM-DD format
    date_errors = []
    dates = []
    for idx, val in enumerate(df['Start Date']):
        try:
            val_str = str(val).strip()
            if val_str in ['', 'nan', 'None']:
                date_errors.append(f"Row {idx + 2}: Empty start date")
                dates.append(None)
            else:
                # Try multiple date formats
                parsed_date = None
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', '%m/%d/%Y']:
                    try:
                        parsed_date = datetime.strptime(val_str, fmt)
                        break
                    except ValueError:
                        continue

                if parsed_date is None:
                    # Try pandas parser as fallback
                    try:
                        parsed_date = pd.to_datetime(val_str, errors='raise')
                        if pd.isna(parsed_date):
                            raise ValueError("Parsed to NaT")
                        parsed_date = parsed_date.to_pydatetime()
                    except:
                        date_errors.append(f"Row {idx + 2}: Invalid date format '{val}' (expected YYYY-MM-DD)")
                        dates.append(None)
                        continue

                dates.append(parsed_date)
        except Exception as e:
            date_errors.append(f"Row {idx + 2}: Error parsing date '{val}': {str(e)}")
            dates.append(None)

    df['Start Date Parsed'] = dates

    if date_errors:
        if len(date_errors) > 5:
            errors.append(f"Date validation errors ({len(date_errors)} total): {date_errors[:5]}... and more")
        else:
            errors.extend(date_errors)

    # Check for duplicate title numbers (same title with multiple allocations is OK, just warn)
    duplicates = df[df.duplicated(subset=['Title Number Normalized'], keep=False)]
    if len(duplicates) > 0:
        dup_titles = duplicates['Title Number'].unique().tolist()
        print(f"  Note: Multiple credit allocations found for titles: {dup_titles}")

    if errors:
        return False, errors, None

    # Return validated dataframe with only valid rows
    valid_df = df[df['Title Number Normalized'].isin(valid_titles_normalized) &
                  df['Amount Parsed'].notna() &
                  df['Start Date Parsed'].notna()].copy()

    return True, [], valid_df


def calculate_credit_expiration_date(start_date: datetime, years_validity: int = 12) -> datetime:
    """
    Calculate the expiration date for exploration credits based on Quebec Mining Act rules.

    Per Section 75 of Quebec's Mining Act, surplus credits can be applied to
    six subsequent terms (each term is 2 years for renewals = 12 years total).

    Args:
        start_date: The allocation/start date of the credits
        years_validity: Number of years credits remain valid (default 12 = 6 terms × 2 years)

    Returns:
        The expiration date for the credits
    """
    return start_date + relativedelta(years=years_validity)


def load_and_apply_credits(excel_path: str, gdf: gpd.GeoDataFrame, output_dir: Optional[str] = None) -> Tuple[gpd.GeoDataFrame, int, float, List[str]]:
    """
    Load credits from Excel file and apply them to the GeoDataFrame.

    Args:
        excel_path: Path to the Excel file with credits
        gdf: GeoDataFrame with claim data
        output_dir: Optional directory to save a log of applied credits

    Returns:
        Tuple of (updated_gdf, number_of_credits_applied, total_credits_added, list_of_credited_title_numbers)
    """
    # Get valid title numbers from the GeoDataFrame
    valid_titles = gdf['title_no'].astype(str).tolist()

    # Validate the Excel file
    print(f"\nValidating credits Excel file: {excel_path}")
    is_valid, errors, credits_df = validate_credits_excel(excel_path, valid_titles)

    if not is_valid:
        print("\n*** VALIDATION FAILED ***")
        for error in errors:
            print(f"  ERROR: {error}")
        raise ValueError("Credits Excel file validation failed. Please fix the errors and try again.")

    print(f"  Validation passed: {len(credits_df)} valid credit entries found")

    # Apply credits to the GeoDataFrame
    credits_applied = 0
    total_amount = 0.0
    applied_credits_log = []
    credited_title_numbers = set()  # Track unique title numbers that received credits

    for _, row in credits_df.iterrows():
        title_normalized = row['Title Number Normalized']
        amount = row['Amount Parsed']
        start_date = row['Start Date Parsed']

        # Find matching claims in GeoDataFrame
        mask = gdf['title_no'].apply(normalize_string) == title_normalized
        matching_indices = gdf[mask].index

        if len(matching_indices) == 0:
            print(f"  Warning: Title {row['Title Number']} not found in GeoDataFrame (skipping)")
            continue

        # Calculate expiration date (12 years = 6 terms × 2 years per Quebec Mining Act)
        expiration_date = calculate_credit_expiration_date(start_date)

        for idx in matching_indices:
            # Add the new credit to the claim's credit_expirations list
            new_credit = {
                'date': pd.Timestamp(expiration_date),
                'amount': round(amount, 2)
            }

            # Get current credit expirations
            current_expirations = gdf.at[idx, 'credit_expirations']
            if not isinstance(current_expirations, list):
                current_expirations = []

            # Add the new credit
            current_expirations.append(new_credit)

            # Sort by date
            current_expirations = sorted(current_expirations, key=lambda x: x['date'])

            # Update the GeoDataFrame
            gdf.at[idx, 'credit_expirations'] = current_expirations

            # Also update the current_excess_work (surplus)
            gdf.at[idx, 'current_excess_work'] += amount
            gdf.at[idx, 'excess_work'] += amount

            # Log the applied credit
            applied_credits_log.append({
                'title_no': row['Title Number'],
                'project': gdf.at[idx, 'project'],
                'amount': amount,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'expiration_date': expiration_date.strftime('%Y-%m-%d'),
                'applied_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            credits_applied += 1
            total_amount += amount
            credited_title_numbers.add(gdf.at[idx, 'title_no'])  # Track the title number

            print(f"  Applied ${amount:,.2f} to title {row['Title Number']} (expires {expiration_date.strftime('%Y-%m-%d')})")

    print(f"\n  Total: {credits_applied} credits applied, ${total_amount:,.2f} added")

    # Export applied credits log if output directory provided
    if output_dir and applied_credits_log:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, f"applied_exploration_credits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        pd.DataFrame(applied_credits_log).to_csv(log_path, index=False)
        print(f"  Applied credits log saved to: {log_path}")

    return gdf, credits_applied, total_amount, list(credited_title_numbers)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from JSON file or use defaults."""
    config = CONFIG.copy()
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config.update(json.load(f))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_path}: {str(e)}")
    # Create directories
    for dir_key in ['OUTPUT_DIR', 'TEMP_DIR', 'LOG_DIR']:
        os.makedirs(config[dir_key], exist_ok=True)
    return config

def normalize_string(s):
    """Normalize title numbers for matching."""
    s = str(s).strip()
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'^(TN|CDC)', '', s, flags=re.IGNORECASE)
    s = s.lstrip('0') or '0'
    return s

def calculate_preferred_filing_date(expiration_date):
    """Calculate preferred filing date (61 days before expiration)."""
    try:
        exp_date = pd.to_datetime(expiration_date, errors='coerce')
        if pd.isna(exp_date):
            return ''
        return (exp_date - timedelta(days=61)).strftime('%Y-%m-%d')
    except Exception:
        return ''

def categorize_quarter_year(expiration_date):
    """Categorize expiration date into quarter-year."""
    try:
        exp_date = pd.to_datetime(expiration_date, errors='coerce')
        if pd.isna(exp_date):
            return ''
        year = exp_date.year
        month = exp_date.month
        quarter = (month - 1) // 3 + 1
        return f"Q{quarter}-{year}"
    except Exception:
        return ''

def check_and_fix_permissions(file_path):
    """Check and fix file permissions."""
    try:
        if os.path.exists(file_path):
            os.chmod(file_path, stat.S_IWRITE)
        directory = os.path.dirname(file_path)
        if not os.access(directory, os.W_OK):
            os.chmod(directory, stat.S_IWRITE)
        return True
    except Exception as e:
        print(f"Failed to set write permissions for {file_path}: {e}")
        return False

# ==============================================================================
# STEP 1: FILTER AND TRANSLATE MIDLAND DATA
# ==============================================================================

def process_midland_file(midland_path: str, property_csv: str, config: Dict[str, Any]) -> pd.DataFrame:
    """Filter and translate Midland Excel file for Casault claims."""
    print("\n=== Step 1: Processing Midland Data ===")

    if not os.path.exists(midland_path):
        raise FileNotFoundError(f"Midland file not found: {midland_path}")

    # Load property mapping
    claim_to_property = {}
    if os.path.exists(property_csv):
        try:
            property_df = pd.read_csv(property_csv, encoding='latin1', dtype=str, keep_default_na=False)
            if 'CLAIM' in property_df.columns and 'PROPERTY' in property_df.columns:
                property_df['CLAIM'] = property_df['CLAIM'].apply(normalize_string)
                claim_to_property = dict(zip(property_df['CLAIM'], property_df['PROPERTY']))
                print(f"Loaded {len(claim_to_property)} property mappings")
        except Exception as e:
            print(f"Warning: Could not load property mapping: {e}")

    # Read Midland Excel
    expected_columns = list(TRANSLATIONS.keys())
    df = pd.read_excel(midland_path, names=expected_columns, header=0, dtype=str, keep_default_na=False)
    print(f"Read {len(df)} rows from Midland file")

    # Normalize and filter for Casault claims
    if 'No titre' not in df.columns:
        raise ValueError("'No titre' column not found in Midland file")

    df['No titre'] = df['No titre'].apply(normalize_string)
    filtered_df = df[df['No titre'].isin(CASAULT_CLAIM_NUMBERS)].copy()
    print(f"Filtered to {len(filtered_df)} Casault claims")

    # Add Property column
    if claim_to_property:
        filtered_df['Property'] = filtered_df['No titre'].map(claim_to_property).fillna('Unknown')
    else:
        filtered_df['Property'] = 'Unknown'

    # Reorder columns
    cols = filtered_df.columns.tolist()
    cols.remove('Property')
    cols.insert(1, 'Property')
    filtered_df = filtered_df[cols]

    # Rename columns
    filtered_df.columns = [TRANSLATIONS.get(col, col) for col in filtered_df.columns]

    # Apply data translations
    for column, mapping in DATA_TRANSLATIONS.items():
        if column in filtered_df.columns:
            filtered_df[column] = filtered_df[column].replace(mapping)

    # Convert numeric columns
    numeric_columns = ['Polygon Area', 'Area (Ha)', 'Surpluses', 'Required Works', 'Required Rights']
    for col in numeric_columns:
        if col in filtered_df.columns:
            filtered_df[col] = filtered_df[col].str.replace(',', '.', regex=False)
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')

    # Add calculated columns
    if 'Expiration Date' in filtered_df.columns:
        filtered_df['Preferred Filing Date'] = filtered_df['Expiration Date'].apply(calculate_preferred_filing_date)
        filtered_df['Expiration Quarter-Year'] = filtered_df['Expiration Date'].apply(categorize_quarter_year)

    print(f"Processed Midland data: {len(filtered_df)} rows")
    return filtered_df

# ==============================================================================
# STEP 2: PROCESS AND MERGE WALLBRIDGE DATA
# ==============================================================================

def process_wallbridge_file(wallbridge_path: str, property_csv: str, config: Dict[str, Any]) -> pd.DataFrame:
    """Process Wallbridge Excel file."""
    print("\n=== Step 2: Processing Wallbridge Data ===")

    if not os.path.exists(wallbridge_path):
        raise FileNotFoundError(f"Wallbridge file not found: {wallbridge_path}")

    # Load property mapping
    claim_to_property = {}
    if os.path.exists(property_csv):
        try:
            property_df = pd.read_csv(property_csv, encoding='latin1', dtype=str, keep_default_na=False)
            if 'CLAIM' in property_df.columns and 'PROPERTY' in property_df.columns:
                property_df['CLAIM'] = property_df['CLAIM'].apply(normalize_string)
                claim_to_property = dict(zip(property_df['CLAIM'], property_df['PROPERTY']))
        except Exception as e:
            print(f"Warning: Could not load property mapping: {e}")

    # Read Wallbridge Excel
    df = pd.read_excel(wallbridge_path, dtype=str, keep_default_na=False)
    print(f"Read {len(df)} rows from Wallbridge file")

    # Rename columns if needed
    df.columns = [TRANSLATIONS.get(col, col) for col in df.columns]

    # Apply data translations
    for column, mapping in DATA_TRANSLATIONS.items():
        if column in df.columns:
            df[column] = df[column].replace(mapping)

    # Convert numeric columns
    numeric_columns = ['Polygon Area', 'Area (Ha)', 'Surpluses', 'Required Works', 'Required Rights']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Add calculated columns
    if 'Expiration Date' in df.columns:
        df['Preferred Filing Date'] = df['Expiration Date'].apply(calculate_preferred_filing_date)
        df['Expiration Quarter-Year'] = df['Expiration Date'].apply(categorize_quarter_year)

    # Add Property column if Title Number exists
    if 'Title Number' in df.columns and claim_to_property:
        df['Title Number'] = df['Title Number'].apply(normalize_string)
        df['Property'] = df['Title Number'].map(claim_to_property).fillna('Unknown')
    elif 'Property' not in df.columns:
        df['Property'] = 'Unknown'

    # Reorder columns to put Property second if needed
    if 'Property' in df.columns:
        cols = df.columns.tolist()
        if 'Property' in cols:
            cols.remove('Property')
            cols.insert(1, 'Property')
            df = df[cols]

    print(f"Processed Wallbridge data: {len(df)} rows")
    return df

def merge_datasets(midland_df: pd.DataFrame, wallbridge_df: pd.DataFrame, config: Dict[str, Any]) -> str:
    """Merge Midland and Wallbridge datasets and save to CSV."""
    print("\n=== Step 3: Merging Datasets ===")

    # Ensure columns match
    all_columns = set(midland_df.columns) | set(wallbridge_df.columns)
    for col in all_columns:
        if col not in midland_df.columns:
            midland_df[col] = ''
        if col not in wallbridge_df.columns:
            wallbridge_df[col] = ''

    # Align column order
    column_order = midland_df.columns.tolist()
    wallbridge_df = wallbridge_df[column_order]

    # Combine
    combined_df = pd.concat([wallbridge_df, midland_df], ignore_index=True)

    # Deduplicate by Title Number
    if 'Title Number' in combined_df.columns:
        combined_df['Title Number'] = combined_df['Title Number'].apply(normalize_string)
        combined_df = combined_df.drop_duplicates(subset='Title Number', keep='last')
        print(f"Deduplicated to {len(combined_df)} unique claims")

    # Save merged CSV
    output_file = os.path.join(
        config['TEMP_DIR'],
        f"Gestim_Wallbridge_Midland_{datetime.now().strftime('%Y%m%d')}.csv"
    )

    if check_and_fix_permissions(output_file):
        combined_df.to_csv(output_file, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        print(f"Merged dataset saved: {output_file}")
    else:
        raise PermissionError(f"Cannot write to {output_file}")

    return output_file

# ==============================================================================
# STEP 3: REDISTRIBUTION SIMULATION (from original script)
# ==============================================================================

def parse_expirations(exp_str: str) -> List[Dict[str, Any]]:
    """Parse expiration string into sorted list of date-amount dictionaries."""
    if pd.isna(exp_str) or exp_str.strip() == '':
        return []
    entries = exp_str.split(';')
    expirations = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        match = re.match(r'(\d{4}/\d{2}/\d{2})\s*\(([\d,.]+)\s*\$?\)', entry)
        if match:
            date_str, amount_str = match.groups()
            try:
                exp_date = pd.to_datetime(date_str, format='%Y/%m/%d')
                amount = round(float(amount_str.replace(',', '.')), 2)
                if amount > 0:
                    expirations.append({'date': exp_date, 'amount': amount})
            except (ValueError, TypeError):
                print(f"Warning: Could not parse expiration entry: {entry}")
        else:
            print(f"Warning: Invalid expiration format: {entry}")
    expirations.sort(key=lambda x: x['date'])
    return expirations

def format_expirations(exp_list: List[Dict[str, Any]]) -> str:
    """Format expiration list as semicolon-separated string."""
    if not isinstance(exp_list, list):
        return exp_list
    return '; '.join(f"{exp['date'].strftime('%Y/%m/%d')} ({exp['amount']:.2f} $)" for exp in exp_list if exp['amount'] > 0) or ''

def reduce_expirations(exp_list: List[Dict[str, Any]], amount: float) -> List[Dict[str, Any]]:
    """Reduce amount from earliest expirations, returning pulled portions."""
    pulled_portions = []
    used = 0
    i = 0
    while used < amount and i < len(exp_list):
        pull = min(amount - used, exp_list[i]['amount'])
        if pull > 0:
            exp_list[i]['amount'] -= pull
            pulled_portions.append({'date': exp_list[i]['date'], 'amount': pull})
            used += pull
        if exp_list[i]['amount'] <= 0:
            del exp_list[i]
        else:
            i += 1
    return pulled_portions

def get_required_work(term: int, area_ha: float, latitude: float) -> float:
    """Compute required minimum work cost based on Quebec regulation."""
    is_north = latitude > 52
    if is_north:
        if area_ha < 25:
            costs = [48, 160, 320, 480, 640, 750, 1000]
        elif area_ha <= 45:
            costs = [120, 400, 800, 1200, 1600, 1800, 2500]
        else:
            costs = [135, 450, 900, 1350, 1800, 1800, 2500]
    else:
        if area_ha < 25:
            costs = [500, 500, 500, 750, 750, 750, 1200]
        elif area_ha <= 100:
            costs = [1200, 1200, 1200, 1800, 1800, 1800, 2500]
        else:
            costs = [1800, 1800, 1800, 2700, 2700, 2700, 3600]
    idx = min(term - 1, 6)
    return costs[idx]

def load_and_prepare_data(csv_path: str, shp_path: str, outlines_path: str, config: Dict[str, Any]) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load and prepare data from CSV and shapefiles."""
    print("\n=== Step 4: Loading Spatial Data ===")

    # Validate files
    for path, name in [(csv_path, "CSV"), (shp_path, "Shapefile"), (outlines_path, "Outlines")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    # Get projects to include from config
    INCLUDED_PROJECTS = config.get('INCLUDED_PROJECTS', ['CASAULT', 'MARTINIERE', 'FENELON', 'GRASSET', 'HARRI', 'DOIGT'])

    # Load shapefile (don't filter yet - wait until after merge with CSV)
    gdf_claims = gpd.read_file(shp_path)
    print(f"Loaded {len(gdf_claims)} claims from shapefile")

    df_csv = pd.read_csv(csv_path, encoding='utf-8')
    print(f"Loaded {len(df_csv)} claims from CSV")

    # Validate columns
    required_cols = ['Title Number', 'Expiration Date', 'Surpluses', 'Required Works', 'Property',
                     'Area (Ha)', 'Number of Deadlines', 'Number of Renewals', 'Expiration Dates and Amounts']
    missing_cols = [col for col in required_cols if col not in df_csv.columns]
    if missing_cols:
        raise ValueError(f"Missing required CSV columns: {missing_cols}")

    # Normalize title numbers
    gdf_claims['title_no'] = gdf_claims['title_no'].astype(str).str.strip()
    df_csv['Title Number'] = df_csv['Title Number'].astype(str).str.strip()

    # Rename project-like columns in shapefile to avoid conflicts
    for col in ['project', 'Property', 'PROJECT', 'property']:
        if col in gdf_claims.columns:
            gdf_claims = gdf_claims.rename(columns={col: f'{col}_shp'})

    # Map CSV columns
    column_mapping = {
        'Title Number': 'title_no',
        'Expiration Date': 'expiry_date',
        'Surpluses': 'excess_work',
        'Required Works': 'required_work',
        'Property': 'project',
        'Area (Ha)': 'area_ha',
        'Number of Deadlines': 'terms_completed',
        'Number of Renewals': 'renewals_done',
        'Expiration Dates and Amounts': 'credit_expirations_raw',
    }
    df_csv = df_csv.rename(columns=column_mapping)

    # Convert numeric columns
    for col in ['excess_work', 'required_work', 'area_ha']:
        df_csv[col] = (
            df_csv[col].astype(str).str.replace(',', '.')
            .replace(['None', ''], '0').astype(float).round(2)
        )
    for col in ['terms_completed', 'renewals_done']:
        df_csv[col] = df_csv[col].fillna(0).astype(int)

    # Parse expirations
    df_csv['credit_expirations'] = df_csv['credit_expirations_raw'].apply(parse_expirations)

    # Merge
    gdf = gdf_claims.merge(
        df_csv[['title_no', 'expiry_date', 'excess_work', 'required_work', 'project',
                'area_ha', 'terms_completed', 'renewals_done', 'credit_expirations']],
        on='title_no', how='inner', suffixes=('_shp', None)
    )

    if gdf.empty:
        raise ValueError("Merge resulted in empty DataFrame")

    # Ensure project column
    if 'project' not in gdf.columns:
        for col in ['project_shp', 'Property_shp']:
            if col in gdf.columns:
                gdf['project'] = gdf[col]
                break
        else:
            raise KeyError("No project column found")

    # Ensure CRS
    if gdf.crs != "EPSG:2958":
        gdf = gdf.to_crs(epsg=2958)

    # Filter to only include specified projects (keeping CASAULT, MARTINIERE, FENELON, GRASSET, HARRI, DOIGT)
    print(f"Before project filtering: {len(gdf)} claims")
    gdf = gdf[gdf['project'].isin(INCLUDED_PROJECTS)].copy().reset_index(drop=True)
    print(f"After project filtering: {len(gdf)} claims in {INCLUDED_PROJECTS}")

    # Load outlines
    gdf_outlines = gpd.read_file(outlines_path)
    if gdf_outlines.crs != "EPSG:2958":
        gdf_outlines = gdf_outlines.to_crs(epsg=2958)

    # Filter outlines shapefile by project using case-insensitive partial matching
    outlines_project_col = None
    for col in ['project', 'PROJECT', 'Project', 'PROPERTY', 'Property', 'property']:
        if col in gdf_outlines.columns:
            outlines_project_col = col
            break

    if outlines_project_col:
        print(f"Filtering outlines shapefile by project column: {outlines_project_col}")
        # Use case-insensitive partial matching since outlines may have "Casault East", "Casault West" etc.
        # while claims have just "CASAULT"
        def matches_project(outline_proj):
            if pd.isna(outline_proj):
                return False
            outline_proj_upper = str(outline_proj).upper()
            # Check if any included project is contained in the outline project name
            return any(proj.upper() in outline_proj_upper for proj in INCLUDED_PROJECTS)

        gdf_outlines_filtered = gdf_outlines[gdf_outlines[outlines_project_col].apply(matches_project)].copy()
        print(f"Outlines shapefile: {len(gdf_outlines)} total, {len(gdf_outlines_filtered)} after filtering")
        print(f"Matched outline projects: {sorted(gdf_outlines_filtered[outlines_project_col].unique())}")
        gdf_outlines = gdf_outlines_filtered
    else:
        print(f"Warning: No project column found in outlines shapefile. Available columns: {list(gdf_outlines.columns)}")

    print(f"Loaded {len(gdf)} claims for simulation from {len(gdf['project'].unique())} projects")
    return gdf, gdf_outlines

def initialize_simulation(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Initialize simulation columns."""
    gdf['current_excess_work'] = gdf['excess_work']
    gdf['current_required_work'] = gdf['required_work']
    gdf['current_expiry_date'] = pd.to_datetime(gdf['expiry_date'], errors='coerce')
    gdf = gdf.dropna(subset=['current_expiry_date'])
    gdf['active'] = True
    gdf['renewals'] = 0
    gdf['current_term'] = gdf['terms_completed'] + gdf['renewals_done'] + 1
    gdf['final_expiry_date'] = gdf['current_expiry_date'].copy()
    gdf['centroid'] = gdf.geometry.centroid
    gdf_4326 = gdf.to_crs(epsg=4326)
    gdf['latitude'] = gdf_4326['centroid'].y
    return gdf

def precompute_spatial_data(gdf: gpd.GeoDataFrame) -> Tuple[np.ndarray, Any]:
    """Precompute distance matrix and spatial index."""
    points = np.array([list(p.coords[0]) for p in gdf.centroid])
    dist_matrix = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=-1)
    cent_gdf = gpd.GeoDataFrame(geometry=gdf.centroid, crs=gdf.crs)
    cent_gdf.index = gdf.index
    sindex = cent_gdf.sindex
    return dist_matrix, sindex

def run_simulation(gdf: gpd.GeoDataFrame, dist_matrix: np.ndarray, sindex: Any,
                   current_date: date, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run main simulation loop."""
    print("\n=== Step 5: Running Redistribution Simulation ===")

    MAX_DISTANCE = config['MAX_DISTANCE']
    MAX_YEAR = config['MAX_YEAR']
    MAX_RENEWALS = config['MAX_RENEWALS']
    SCORING_MODE = config['SCORING_MODE']
    w1 = config['SCORING_WEIGHTS']['surplus']
    w2 = config['SCORING_WEIGHTS']['distance']
    log_table = []
    unresolved = []

    while True:
        active_claims = gdf[gdf['active']]
        if active_claims.empty:
            break
        next_date = active_claims['current_expiry_date'].min()
        if pd.isna(next_date) or next_date.year > MAX_YEAR:
            gdf.at[active_claims.index, 'active'] = False
            break

        # Apply expirations
        for idx in tqdm(active_claims.index, desc=f"Processing expirations for {next_date}"):
            exp_list = gdf.at[idx, 'credit_expirations']
            to_remove = 0
            new_list = []
            for exp in exp_list:
                if exp['date'] > next_date:
                    new_list.append(exp)
                else:
                    to_remove += exp['amount']
            gdf.at[idx, 'credit_expirations'] = new_list
            gdf.at[idx, 'current_excess_work'] -= to_remove

        # Process expiring claims
        expiring = active_claims[active_claims['current_expiry_date'] == next_date]
        expiring = expiring.assign(deficit=np.maximum(0, expiring['current_required_work'] - expiring['current_excess_work']))
        expiring = expiring.sort_values(by='deficit', ascending=True)

        for idx, row in tqdm(expiring.iterrows(), desc=f"Processing claims expiring on {next_date}", total=len(expiring)):
            if row['renewals'] >= MAX_RENEWALS:
                unresolved.append({
                    'year': next_date.year,
                    'project': row['project'],
                    'deficit_needed': row['current_required_work'] - row['current_excess_work'],
                    'title_no': row['title_no']
                })
                gdf.at[idx, 'active'] = False
                continue

            excess = gdf.at[idx, 'current_excess_work']
            required = gdf.at[idx, 'current_required_work']
            deficit = required - excess

            if deficit <= 0:
                # Self-renew
                original_exp = copy.deepcopy(gdf.at[idx, 'credit_expirations'])
                pulled_portions = reduce_expirations(gdf.at[idx, 'credit_expirations'], required)
                gdf.at[idx, 'current_excess_work'] -= required
                gdf.at[idx, 'current_expiry_date'] += relativedelta(years=2)
                gdf.at[idx, 'renewals'] += 1
                gdf.at[idx, 'current_term'] += 1
                gdf.at[idx, 'current_required_work'] = get_required_work(gdf.at[idx, 'current_term'], row['area_ha'], row['latitude'])
                gdf.at[idx, 'final_expiry_date'] = gdf.at[idx, 'current_expiry_date']
                log_table.append({
                    'action_type': 'renewal',
                    'title_no': row['title_no'],
                    'renewal_date': next_date,
                    'renewal_year': next_date.year,
                    'renewal_amount': required,
                    'pulled_expirations': format_expirations(pulled_portions),
                    'original_expirations': format_expirations(original_exp),
                    'updated_expirations': format_expirations(gdf.at[idx, 'credit_expirations'])
                })
                continue

            # Find donors
            credits_needed = deficit
            buffer_geo = row['centroid'].buffer(MAX_DISTANCE)
            candidates_pos = sindex.query(buffer_geo, predicate='intersects')
            nearby = gdf.iloc[candidates_pos]
            nearby = nearby[
                (nearby['active']) &
                (nearby.index != idx) &
                (nearby['current_excess_work'] > nearby['current_required_work'])
            ]

            # Casault isolation
            if row['project'] == 'CASAULT':
                nearby = nearby[nearby['project'] == 'CASAULT']
            else:
                nearby = nearby[nearby['project'] != 'CASAULT']

            if not nearby.empty:
                nearby = nearby.assign(
                    distance=dist_matrix[idx][nearby.index],
                    available=nearby['current_excess_work'] - nearby['current_required_work']
                )

                if SCORING_MODE == 'earliest_expiry':
                    nearby = nearby.assign(
                        score=[min((exp['date'] for exp in gdf.at[idx, 'credit_expirations']), default=pd.Timestamp.max).timestamp() for idx in nearby.index]
                    )
                    nearby = nearby.sort_values(by='score', ascending=True)
                else:
                    min_avail = nearby['available'].min()
                    max_avail = nearby['available'].max()
                    min_dist = nearby['distance'].min()
                    max_dist = nearby['distance'].max()
                    if max_avail > min_avail:
                        norm_surplus = (nearby['available'] - min_avail) / (max_avail - min_avail)
                    else:
                        norm_surplus = pd.Series(1.0, index=nearby.index)
                    if max_dist > min_dist:
                        log_dist = np.log1p(nearby['distance'])
                        log_min = np.log1p(min_dist)
                        log_max = np.log1p(max_dist)
                        norm_distance = (log_dist - log_min) / (log_max - log_min)
                    else:
                        norm_distance = pd.Series(0.0, index=nearby.index)
                    nearby = nearby.assign(score=(w1 * norm_surplus) + (w2 * (1 - norm_distance)))
                    nearby = nearby.sort_values(by='score', ascending=False)

                for donor_idx, donor in nearby.iterrows():
                    avail = gdf.at[donor_idx, 'current_excess_work'] - gdf.at[donor_idx, 'current_required_work']
                    pull = min(avail, credits_needed)
                    if pull > 0:
                        original_exp_donor = copy.deepcopy(gdf.at[donor_idx, 'credit_expirations'])
                        pulled_portions = reduce_expirations(gdf.at[donor_idx, 'credit_expirations'], pull)
                        gdf.at[donor_idx, 'current_excess_work'] -= pull
                        gdf.at[idx, 'credit_expirations'].extend(pulled_portions)
                        gdf.at[idx, 'credit_expirations'] = sorted(gdf.at[idx, 'credit_expirations'], key=lambda d: d['date'])
                        gdf.at[idx, 'current_excess_work'] += pull
                        log_table.append({
                            'action_type': 'redistribution',
                            'insufficient_title_no': row['title_no'],
                            'recipient_project': row['project'],
                            'expiry_date': next_date,
                            'redistribution_year': next_date.year,
                            'deficit_needed': pull,
                            'donor_title_no': donor['title_no'],
                            'donor_project': donor['project'],
                            'donor_original_excess': donor['current_excess_work'],
                            'donor_expiration_date': donor['current_expiry_date'],
                            'credits_pulled': pull,
                            'donor_new_excess_work': gdf.at[donor_idx, 'current_excess_work'],
                            'pulled_expirations': format_expirations(pulled_portions),
                            'original_expirations_donor': format_expirations(original_exp_donor),
                            'updated_expirations_donor': format_expirations(gdf.at[donor_idx, 'credit_expirations']),
                            'distance_m': donor['distance']
                        })
                        credits_needed -= pull
                    if credits_needed <= 0:
                        break

            if credits_needed <= 0:
                # Renew after receiving credits
                original_exp = copy.deepcopy(gdf.at[idx, 'credit_expirations'])
                pulled_portions = reduce_expirations(gdf.at[idx, 'credit_expirations'], gdf.at[idx, 'current_required_work'])
                gdf.at[idx, 'current_excess_work'] -= gdf.at[idx, 'current_required_work']
                gdf.at[idx, 'current_expiry_date'] += relativedelta(years=2)
                gdf.at[idx, 'renewals'] += 1
                gdf.at[idx, 'current_term'] += 1
                gdf.at[idx, 'current_required_work'] = get_required_work(gdf.at[idx, 'current_term'], row['area_ha'], row['latitude'])
                gdf.at[idx, 'final_expiry_date'] = gdf.at[idx, 'current_expiry_date']
                log_table.append({
                    'action_type': 'renewal',
                    'title_no': row['title_no'],
                    'renewal_date': next_date,
                    'renewal_year': next_date.year,
                    'renewal_amount': gdf.at[idx, 'current_required_work'],
                    'pulled_expirations': format_expirations(pulled_portions),
                    'original_expirations': format_expirations(original_exp),
                    'updated_expirations': format_expirations(gdf.at[idx, 'credit_expirations'])
                })
            else:
                unresolved.append({
                    'year': next_date.year,
                    'project': row['project'],
                    'deficit_needed': credits_needed,
                    'title_no': row['title_no']
                })
                gdf.at[idx, 'active'] = False

    return log_table, unresolved

def export_results(log_table: List[Dict[str, Any]], unresolved: List[Dict[str, Any]],
                   gdf: gpd.GeoDataFrame, output_dir: str, current_date: date) -> None:
    """Export simulation results."""
    print("\n=== Step 6: Exporting Results ===")

    os.makedirs(output_dir, exist_ok=True)

    # Export logs
    pd.DataFrame(log_table).to_csv(os.path.join(output_dir, 'full_redistribution_log.csv'), index=False)
    print(f"Exported redistribution log")

    # Export unresolved
    if unresolved:
        unresolved_df = pd.DataFrame(unresolved)
        grouped = unresolved_df.groupby(['year', 'project'])['deficit_needed'].sum().reset_index(name='total_money_needed')
        grouped.to_csv(os.path.join(output_dir, 'unresolved_credits_by_year_property.csv'), index=False)

        current_year = current_date.year
        max_year = current_year + 20
        grouped_filtered = grouped[grouped['year'] <= max_year]
        if not grouped_filtered.empty:
            pivot = grouped_filtered.pivot(index='year', columns='project', values='total_money_needed').fillna(0)
            pivot['Total'] = pivot.sum(axis=1)
            pivot.to_csv(os.path.join(output_dir, 'required_spend_pivot.csv'))
        print(f"Exported unresolved credits")

    # Export life expectancy
    gdf['days_of_life'] = (gdf['final_expiry_date'] - pd.Timestamp(current_date)).dt.days
    gdf['years_of_life'] = gdf['days_of_life'] / 365.25
    gdf[['title_no', 'project', 'expiry_date', 'final_expiry_date', 'renewals', 'years_of_life']].to_csv(
        os.path.join(output_dir, 'claims_years_of_life.csv'), index=False
    )
    print(f"Exported life expectancy data")

    # Export expiration history
    expiration_history = []
    for idx, row in gdf.iterrows():
        expiration_history.append({
            'title_no': row['title_no'],
            'project': row['project'],
            'final_expiry_date': row['final_expiry_date'],
            'credit_expirations': format_expirations(row['credit_expirations'])
        })
    pd.DataFrame(expiration_history).to_csv(os.path.join(output_dir, 'claim_expiration_history.csv'), index=False)

    # Summary report
    report = [
        f"Simulation Summary Report",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Claims Analyzed: {len(gdf)}",
        f"Average Years of Life: {gdf['years_of_life'].mean():.2f}",
        f"Minimum Years of Life: {gdf['years_of_life'].min():.2f}",
        f"Maximum Years of Life: {gdf['years_of_life'].max():.2f}",
        f"Total Credits Redistributed: {sum(entry['credits_pulled'] for entry in log_table if entry['action_type'] == 'redistribution'):.2f}",
        f"Claims Lapsed: {len(unresolved)}"
    ]
    with open(os.path.join(output_dir, 'simulation_report.txt'), 'w') as f:
        f.write('\n'.join(report))
    print(f"Exported summary report")

def plot_interactive_map(gdf: gpd.GeoDataFrame, log_table: List[Dict[str, Any]],
                         unresolved: List[Dict[str, Any]], gdf_outlines: gpd.GeoDataFrame,
                         output_dir: str) -> None:
    """Create interactive map."""
    print("\n=== Step 7: Creating Interactive Map ===")

    gdf_4326 = gdf.to_crs(epsg=4326)
    gdf_outlines_4326 = gdf_outlines.to_crs(epsg=4326)
    gdf_centroids_4326 = gpd.GeoDataFrame(geometry=gdf['centroid'], crs='EPSG:2958').to_crs(epsg=4326)

    m = folium.Map(
        location=[gdf_centroids_4326.geometry.y.mean(), gdf_centroids_4326.geometry.x.mean()],
        zoom_start=10,
        tiles='Stamen Terrain',
        attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL'
    )
    Draw(export=True).add_to(m)

    # Define colors
    gdf['life_category'] = gdf['final_expiry_date'].apply(lambda x: str(x.year))
    unique_years = sorted(gdf['life_category'].unique())
    n_years = len(unique_years)
    if n_years > 0:
        colors = mcolors.LinearSegmentedColormap.from_list(
            'hot_to_cold', ['red', 'orange', 'yellow', 'green', 'blue'], N=n_years
        )(np.linspace(0, 1, n_years))
        cat_colors = {year: mcolors.to_hex(colors[i]) for i, year in enumerate(unique_years)}
    else:
        cat_colors = {}

    # Add claims
    claims_group = folium.FeatureGroup(name='Claims', show=True)
    for idx, row in gdf_4326.iterrows():
        tooltip = f"""
        Title No: {row['title_no']}<br>
        Project: {row['project']}<br>
        Final Expiry Date: {row['final_expiry_date'].strftime('%Y/%m/%d')}<br>
        Years of Life: {row['years_of_life']:.2f}<br>
        Credit Expirations: {format_expirations(row['credit_expirations'])}
        """
        folium.GeoJson(
            row.geometry,
            style_function=lambda x, year=row['life_category']: {
                'fillColor': cat_colors.get(year, 'gray'),
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.7
            },
            tooltip=tooltip
        ).add_to(claims_group)
    claims_group.add_to(m)

    # Add outlines
    outlines_group = folium.FeatureGroup(name='Property Outlines', show=True)
    folium.GeoJson(
        gdf_outlines_4326,
        style_function=lambda x: {'color': 'black', 'weight': 2, 'fillOpacity': 0}
    ).add_to(outlines_group)
    outlines_group.add_to(m)

    # Add flows
    flow_group = folium.FeatureGroup(name='Redistribution Flows', show=False)
    for entry in log_table:
        if entry['action_type'] == 'redistribution' and entry['donor_title_no'] is not None:
            donor = gdf_4326[gdf_4326['title_no'] == entry['donor_title_no']]
            recipient = gdf_4326[gdf_4326['title_no'] == entry['insufficient_title_no']]
            if not donor.empty and not recipient.empty:
                donor = donor.iloc[0]
                recipient = recipient.iloc[0]
                donor_centroid = gdf_centroids_4326[gdf_4326.index == donor.name].geometry.iloc[0]
                recipient_centroid = gdf_centroids_4326[gdf_4326.index == recipient.name].geometry.iloc[0]
                folium.PolyLine(
                    locations=[(donor_centroid.y, donor_centroid.x), (recipient_centroid.y, recipient_centroid.x)],
                    color='gray',
                    weight=2,
                    opacity=0.5,
                    tooltip=f"Credits Pulled: {entry['credits_pulled']:.2f}<br>Distance: {entry['distance_m']:.2f} m"
                ).add_to(flow_group)
    flow_group.add_to(m)

    folium.LayerControl().add_to(m)

    # Legend
    legend_html = '''
    <div style="position: fixed;
    bottom: 50px; left: 50px; width: 150px; height: auto;
    border:2px solid grey; z-index:9999; font-size:14px;
    background-color:white; padding: 10px;">
    <b>Lapse Year Legend</b> <br>
    '''
    for year in unique_years:
        color = cat_colors.get(year, '#808080')
        legend_html += f'<i style="background:{color};width:20px;height:20px;float:left;margin-right:10px;"></i> {year}<br>'
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(os.path.join(output_dir, 'interactive_gsm_claims_map.html'))
    print(f"Exported interactive map")

def plot_results(gdf: gpd.GeoDataFrame, gdf_outlines: gpd.GeoDataFrame, output_dir: str,
                 credited_titles: Optional[List[str]] = None) -> None:
    """Generate static visualization.

    Args:
        gdf: GeoDataFrame with claim data
        gdf_outlines: GeoDataFrame with property outlines
        output_dir: Directory to save the output map
        credited_titles: Optional list of title numbers that received exploration credits
                        (will be highlighted with neon purple border)
    """
    print("\n=== Step 8: Creating Static Map ===")

    # Use non-interactive backend for thread safety
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    gdf['life_category'] = gdf['final_expiry_date'].apply(lambda x: str(x.year))
    unique_years = sorted(gdf['life_category'].unique())
    n_years = len(unique_years)
    if n_years > 0:
        colors = mcolors.LinearSegmentedColormap.from_list(
            'hot_to_cold', ['red', 'orange', 'yellow', 'green', 'blue'], N=n_years
        )(np.linspace(0, 1, n_years))
        cat_colors = {year: colors[i] for i, year in enumerate(unique_years)}
    else:
        cat_colors = {}

    # Create figure with more width to accommodate legend on the right
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot claims by lapse year
    for cat in unique_years:
        gdf_cat = gdf[gdf['life_category'] == cat]
        if gdf_cat.empty:
            continue
        gdf_cat.plot(
            ax=ax,
            color=cat_colors[cat],
            edgecolor='black',
            linewidth=0.3,
            alpha=0.7,
            label=cat
        )

    # Plot neon purple border around claims that received exploration credits
    has_credited_claims = False
    if credited_titles and len(credited_titles) > 0:
        # Filter to claims that received credits
        gdf_credited = gdf[gdf['title_no'].isin(credited_titles)]
        if not gdf_credited.empty:
            has_credited_claims = True
            # Plot only the border (no fill) with neon purple color
            gdf_credited.plot(
                ax=ax,
                facecolor='none',
                edgecolor='#BF00FF',  # Neon purple
                linewidth=1.5,
                alpha=1.0,
                zorder=5  # Above regular claims but below property outlines
            )
            print(f"  Highlighted {len(gdf_credited)} claims with added exploration credits")

    # Plot property outlines ON TOP with black borders
    gdf_outlines.plot(
        ax=ax,
        facecolor='none',
        edgecolor='black',
        linewidth=1.5,
        zorder=10  # Ensure outlines are drawn on top
    )

    # Add professional north arrow
    from matplotlib.patches import FancyArrowPatch
    arrow_x, arrow_y = 0.05, 0.92
    arrow_length = 0.05

    # Create arrow with better styling
    arrow = FancyArrowPatch(
        (arrow_x, arrow_y), (arrow_x, arrow_y + arrow_length),
        transform=ax.transAxes,
        arrowstyle='simple',
        mutation_scale=30,
        facecolor='black',
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(arrow)

    # Add 'N' label below arrow
    ax.text(arrow_x, arrow_y - 0.01, 'N',
            transform=ax.transAxes,
            fontsize=18, fontweight='bold',
            ha='center', va='top')

    # Add scale bar
    scalebar = AnchoredSizeBar(
        ax.transData,
        1000,
        '1 km',
        loc='lower right',
        pad=0.5,
        color='black',
        frameon=False,
        size_vertical=100,
        fontproperties={'size': 10, 'weight': 'bold'}
    )
    ax.add_artist(scalebar)

    # Set title and labels
    plt.title('Projected Lapse Years for GSM Claims with Ongoing Redistributions - UTM Zone 17N',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Easting (meters)', fontsize=12, fontweight='bold')
    plt.ylabel('Northing (meters)', fontsize=12, fontweight='bold')

    # Remove grid lines
    ax.grid(False)

    # Format axis tick labels to remove decimals and commas
    ax.ticklabel_format(style='plain', axis='both', useOffset=False)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f'{int(y)}'))

    # Create legend with lapse years, credited claims, and property outlines
    legend_elements = [
        Patch(facecolor=cat_colors[year], edgecolor='black', alpha=0.7, label=year)
        for year in unique_years
    ]
    # Add credited claims legend entry if applicable
    if has_credited_claims:
        legend_elements.append(Patch(facecolor='none', edgecolor='#BF00FF', linewidth=1.5, label='Added Credits'))
    legend_elements.append(Patch(facecolor='none', edgecolor='black', linewidth=1.5, label='Property Outlines'))

    # Place legend outside the plot area on the right
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=10, title='Lapse Year', title_fontsize=12, frameon=True,
              fancybox=True, shadow=True)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(os.path.join(output_dir, 'gsm_claims_lapse_years_map.png'), dpi=300, bbox_inches='tight')
    print(f"Exported static map")
    plt.close()

def plot_summary_by_project(gdf: gpd.GeoDataFrame, unresolved: List[Dict[str, Any]], output_dir: str) -> None:
    """Generate summary bar plot."""
    print("\n=== Step 9: Creating Summary Plot ===")

    # Use non-interactive backend for thread safety
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if 'life_category' not in gdf.columns:
        gdf['life_category'] = gdf['final_expiry_date'].apply(lambda x: str(x.year))

    plt.figure(figsize=(12, 6))
    order = sorted(gdf['life_category'].unique())
    ax = sns.countplot(data=gdf, x='life_category', hue='project', order=order)

    if unresolved:
        unresolved_df = pd.DataFrame(unresolved)
        deficit_grouped = unresolved_df.groupby(['year', 'project'])['deficit_needed'].sum().reset_index(name='total_deficit')
        deficit_grouped['year'] = deficit_grouped['year'].astype(str)

        bars = ax.patches
        bar_data = []
        for bar in bars:
            bar_data.append({
                'x': bar.get_x() + bar.get_width() / 2,
                'height': bar.get_height(),
                'year': order[int(bar.get_x() + bar.get_width() / 2)] if int(bar.get_x() + bar.get_width() / 2) < len(order) else None,
                'project': bar.get_label() if hasattr(bar, 'get_label') else None
            })

        for bar in bar_data:
            year = bar['year']
            project = bar['project']
            if year and project and project in deficit_grouped['project'].values:
                deficit_row = deficit_grouped[(deficit_grouped['year'] == year) & (deficit_grouped['project'] == project)]
                if not deficit_row.empty:
                    deficit = deficit_row['total_deficit'].iloc[0]
                    if deficit >= 1_000_000:
                        deficit_text = f"${deficit/1_000_000:.1f}M"
                    elif deficit >= 1_000:
                        deficit_text = f"${deficit/1_000:.1f}K"
                    else:
                        deficit_text = f"${deficit:.0f}"
                    ax.text(
                        bar['x'],
                        bar['height'] + 0.1,
                        deficit_text,
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        rotation=45
                    )

    plt.title('Number of Lapsed Claims by Project and Year')
    plt.xlabel('Lapse Year')
    plt.ylabel('Number of Lapsed Claims')
    plt.legend(title='Project')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'claims_by_project.png'), dpi=300, bbox_inches='tight')
    print(f"Exported summary plot")
    plt.close()

def generate_pdf_report(gdf: gpd.GeoDataFrame, log_table: List[Dict[str, Any]],
                        output_dir: str, current_date: date) -> None:
    """Generate professional PDF report with black border (matches GUI app style)."""
    if not REPORTLAB_AVAILABLE:
        print("\n⚠ Warning: ReportLab not available. Skipping PDF generation.")
        print("Install with: pip install reportlab")
        return

    print("\n=== Step 10: Generating PDF Report ===")

    try:
        output_dir_path = Path(output_dir)
        pdf_filename = output_dir_path / f"Redistribution_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        # Custom page template with black border
        class BorderedDocTemplate(BaseDocTemplate):
            def __init__(self, filename, **kwargs):
                BaseDocTemplate.__init__(self, filename, **kwargs)

            def handle_pageBegin(self):
                BaseDocTemplate.handle_pageBegin(self)
                # Draw black border around page
                self.canv.setStrokeColor(rl_colors.black)
                self.canv.setLineWidth(2)
                margin = 0.25 * inch
                self.canv.rect(margin, margin,
                               letter[0] - 2*margin,
                               letter[1] - 2*margin)

        # Create PDF with bordered template
        doc = BorderedDocTemplate(str(pdf_filename), pagesize=letter,
                                 leftMargin=0.75*inch, rightMargin=0.75*inch,
                                 topMargin=0.75*inch, bottomMargin=0.75*inch)

        # Define frame and page template
        frame = Frame(doc.leftMargin, doc.bottomMargin,
                     doc.width, doc.height, id='normal')
        template = PageTemplate(id='bordered', frames=frame,
                               onPage=lambda canvas, doc: None)
        doc.addPageTemplates([template])

        story = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=rl_colors.HexColor(ModernStyle.PRIMARY),
            spaceAfter=12,
            alignment=TA_CENTER
        )

        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=rl_colors.HexColor(ModernStyle.SECONDARY),
            spaceAfter=8,
            spaceBefore=8
        )

        # ========== PAGE 1 ==========
        # Title
        story.append(Paragraph("MINING CLAIM REDISTRIBUTION ANALYSIS", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}",
                             ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER)))
        story.append(Spacer(1, 0.2*inch))

        # Summary Statistics Table
        story.append(Paragraph("SUMMARY STATISTICS", heading_style))

        # Calculate credit statistics from redistribution log
        full_log_path = output_dir_path / 'full_redistribution_log.csv'
        total_credits_available = 0
        credits_used_renewals = 0
        credits_redistributed = 0

        if full_log_path.exists():
            try:
                log_df = pd.read_csv(full_log_path)

                # Helper function to parse credit values
                def parse_credits(exp_str):
                    """Parse credits from format like '2029/04/09 (12946.51 $)'"""
                    if pd.isna(exp_str) or exp_str == '':
                        return 0
                    total = 0
                    for part in str(exp_str).split(';'):
                        if '(' in part and '$' in part:
                            try:
                                credit_str = part.split('(')[1].split('$')[0].strip()
                                total += float(credit_str)
                            except:
                                pass
                    return total

                # Calculate total credits available from original_expirations
                log_df['original_credits'] = log_df['original_expirations'].apply(parse_credits)
                total_credits_available = log_df['original_credits'].sum()

                # Calculate credits used for renewals
                renewal_rows = log_df[log_df['action_type'] == 'renewal']
                if not renewal_rows.empty and 'renewal_amount' in renewal_rows.columns:
                    credits_used_renewals = renewal_rows['renewal_amount'].sum()

                # Calculate credits redistributed
                redistribution_rows = log_df[log_df['action_type'] == 'redistribution']
                if not redistribution_rows.empty and 'credits_pulled' in redistribution_rows.columns:
                    credits_redistributed = redistribution_rows['credits_pulled'].sum()

            except Exception as e:
                print(f"Warning: Could not calculate credit statistics: {e}")

        credits_unused = total_credits_available - credits_used_renewals - credits_redistributed

        summary_data = [
            ['Total Claims', 'Avg Years Life', 'Total Credits Available', 'Credits Unused'],
            [str(len(gdf)), f"{gdf['years_of_life'].mean():.1f}",
             f"${total_credits_available:,.0f}", f"${credits_unused:,.0f}"],
            ['Credits for Renewals', 'Credits Redistributed', '% Unused', '% Renewals'],
            [f"${credits_used_renewals:,.0f}", f"${credits_redistributed:,.0f}",
             f"{(credits_unused/total_credits_available*100):.1f}%" if total_credits_available > 0 else "0%",
             f"{(credits_used_renewals/total_credits_available*100):.1f}%" if total_credits_available > 0 else "0%"]
        ]

        summary_table = Table(summary_data, colWidths=[1.7*inch, 1.7*inch, 1.7*inch, 1.7*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor(ModernStyle.SECONDARY)),
            ('BACKGROUND', (0, 2), (-1, 2), rl_colors.HexColor(ModernStyle.SECONDARY)),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
            ('TEXTCOLOR', (0, 2), (-1, 2), rl_colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 2), (-1, 2), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 1), (-1, 1), rl_colors.HexColor(ModernStyle.LIGHT)),
            ('BACKGROUND', (0, 3), (-1, 3), rl_colors.HexColor(ModernStyle.LIGHT)),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor(ModernStyle.DARK))
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.15*inch))

        # Projected Lapse Years Map (larger)
        story.append(Paragraph("PROJECTED LAPSE YEARS MAP", heading_style))
        map_file = output_dir_path / 'gsm_claims_lapse_years_map.png'
        if map_file.exists():
            from PIL import Image as PILImage
            pil_img = PILImage.open(str(map_file))
            img_width, img_height = pil_img.size
            aspect = img_height / img_width

            # Larger map for page 1
            max_width = 7.0*inch
            map_height = max_width * aspect

            if map_height > 5.0*inch:
                map_height = 5.0*inch
                max_width = map_height / aspect

            img = RLImage(str(map_file), width=max_width, height=map_height)
            story.append(img)
        story.append(Spacer(1, 0.15*inch))

        # Generate Charts Programmatically
        # Use non-interactive backend
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Chart 1: Claims by Project and Year (on Page 1)
        story.append(Paragraph("NUMBER OF LAPSED CLAIMS BY PROJECT AND YEAR", heading_style))

        fig1, ax1 = plt.subplots(figsize=(10, 4))
        if 'life_category' not in gdf.columns:
            gdf['life_category'] = gdf['final_expiry_date'].apply(lambda x: str(x.year))

        order = sorted(gdf['life_category'].unique())
        sns.countplot(data=gdf, x='life_category', hue='project', order=order, ax=ax1, palette='tab10')

        ax1.set_xlabel('Lapse Year', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Number of Lapsed Claims', fontweight='bold', fontsize=11)
        ax1.legend(title='Project', fontsize=9, loc='upper left')
        ax1.grid(axis='y', alpha=0.2, linestyle='--')
        ax1.tick_params(axis='x', rotation=45, labelsize=9)
        fig1.tight_layout()

        # Save to bytes and add to PDF
        buf1 = BytesIO()
        fig1.savefig(buf1, format='png', dpi=150, bbox_inches='tight')
        buf1.seek(0)
        plt.close(fig1)

        img1 = RLImage(buf1, width=7.0*inch, height=2.8*inch)
        story.append(img1)

        # Page break after first chart to start page 2
        story.append(PageBreak())

        # ========== PAGE 2 ==========
        # Chart 2: Average Claim Life by Project
        story.append(Paragraph("AVERAGE CLAIM LIFE BY PROJECT", heading_style))

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        project_stats = gdf.groupby('project')['years_of_life'].mean().sort_values()

        colors_list = ['#3498DB', '#16A085', '#27AE60', '#F39C12', '#2C3E50', '#E74C3C']
        bars = ax2.barh(project_stats.index, project_stats.values,
                      color=colors_list[:len(project_stats)], edgecolor='#34495E', linewidth=1.5)

        ax2.set_xlabel('Average Years of Life', fontweight='bold', fontsize=11)
        ax2.set_ylabel('Project', fontweight='bold', fontsize=11)
        ax2.grid(axis='x', alpha=0.2, linestyle='--')

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2, f' {width:.1f}',
                   ha='left', va='center', fontweight='bold', fontsize=9)

        fig2.tight_layout()

        # Save to bytes and add to PDF
        buf2 = BytesIO()
        fig2.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
        buf2.seek(0)
        plt.close(fig2)

        img2 = RLImage(buf2, width=7.0*inch, height=2.8*inch)
        story.append(img2)
        story.append(Spacer(1, 0.2*inch))

        # Pivot Table (Years as rows, Projects as columns)
        story.append(Paragraph("REQUIRED SPENDING BY YEAR AND PROJECT", heading_style))

        gdf['lapse_year'] = gdf['final_expiry_date'].apply(lambda x: x.year)
        pivot_data = gdf.groupby(['lapse_year', 'project']).size().reset_index(name='claims_count')
        RENEWAL_COST_PER_CLAIM = 2500  # Actual cost per claim renewal
        pivot_data['required_spending'] = pivot_data['claims_count'] * RENEWAL_COST_PER_CLAIM

        # Years as rows, projects as columns
        pivot = pivot_data.pivot(index='lapse_year', columns='project', values='required_spending')
        pivot = pivot.fillna(0)

        # Get project list and limit to first 10 years
        projects = list(pivot.columns)
        years = sorted([int(y) for y in pivot.index])[:10]

        # Create table with years as rows
        pivot_table_data = [['Year'] + projects + ['TOTAL']]

        for year in years:
            row = [str(year)]
            year_total = 0
            for project in projects:
                val = pivot.loc[year, project] if year in pivot.index and project in pivot.columns else 0
                row.append(f"${val:,.0f}")
                year_total += val
            row.append(f"${year_total:,.0f}")
            pivot_table_data.append(row)

        # Add totals row (total per project)
        totals_row = ['TOTAL']
        grand_total = 0
        for project in projects:
            if project in pivot.columns:
                project_total = pivot[project].sum()
                totals_row.append(f"${project_total:,.0f}")
                grand_total += project_total
            else:
                totals_row.append("$0")
        totals_row.append(f"${grand_total:,.0f}")
        pivot_table_data.append(totals_row)

        col_widths = [0.7*inch] + [1.0*inch] * len(projects) + [1.0*inch]
        pivot_table = Table(pivot_table_data, colWidths=col_widths)
        pivot_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor(ModernStyle.SECONDARY)),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BACKGROUND', (0, 1), (-1, -2), rl_colors.HexColor(ModernStyle.LIGHT)),
            ('BACKGROUND', (0, -1), (-1, -1), rl_colors.HexColor('#E0E0E0')),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor(ModernStyle.DARK))
        ]))
        story.append(pivot_table)
        story.append(Spacer(1, 0.15*inch))

        # Upcoming Concerns
        story.append(Paragraph("UPCOMING CONCERNS (1-2 YEARS)", heading_style))
        current_year = current_date.year
        near_term = gdf[gdf['final_expiry_date'].dt.year <= current_year + 2]

        if len(near_term) > 0:
            concern_text = f"""<b>{len(near_term)} claims</b> expire within {current_year}-{current_year+2}. Projects at risk: """
            proj_list = []
            for proj in near_term['project'].unique():
                proj_claims = near_term[near_term['project'] == proj]
                proj_list.append(f"{proj} ({len(proj_claims)})")
            concern_text += ", ".join(proj_list) + "."
            story.append(Paragraph(concern_text, styles['Normal']))
        else:
            story.append(Paragraph("No critical concerns for next 1-2 years.", styles['Normal']))

        # Build PDF
        doc.build(story)

        print(f"PDF report generated: {pdf_filename}")

    except Exception as e:
        print(f"Error generating PDF: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Integrated Mining Claim Redistribution System')
    parser.add_argument('--midland-xlsx', type=str, help='Path to Midland Excel file')
    parser.add_argument('--wallbridge-xlsx', type=str, help='Path to Wallbridge Excel file')
    parser.add_argument('--property-csv', type=str, help='Path to Property_to_Claim.csv')
    parser.add_argument('--shp-path', type=str, help='Path to claims shapefile')
    parser.add_argument('--outlines-shp', type=str, help='Path to outlines shapefile')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--max-distance', type=float, help='Maximum redistribution distance (m)')
    parser.add_argument('--scoring-mode', type=str, choices=['earliest_expiry', 'distance_surplus'], help='Donor scoring mode')
    parser.add_argument('--current-date', type=str, help='Simulation start date (YYYY-MM-DD)')
    parser.add_argument('--included-projects', type=str, nargs='+', help='List of projects to include (e.g., CASAULT MARTINIERE FENELON)')
    parser.add_argument('--config-path', type=str, help='Path to JSON config file')
    parser.add_argument('--credits-xlsx', type=str, help='Path to Excel file with exploration program credits to add before simulation')
    parser.add_argument('--skip-credits-prompt', action='store_true', help='Skip the prompt asking whether to add credits')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config_path)

    # Override with command-line arguments
    if args.midland_xlsx:
        config['MIDLAND_XLSX'] = args.midland_xlsx
    if args.wallbridge_xlsx:
        config['WALLBRIDGE_XLSX'] = args.wallbridge_xlsx
    if args.property_csv:
        config['PROPERTY_CSV'] = args.property_csv
    if args.shp_path:
        config['SHP_PATH'] = args.shp_path
    if args.outlines_shp:
        config['OUTLINES_SHP'] = args.outlines_shp
    if args.output_dir:
        config['OUTPUT_DIR'] = args.output_dir
    if args.max_distance:
        config['MAX_DISTANCE'] = args.max_distance
    if args.scoring_mode:
        config['SCORING_MODE'] = args.scoring_mode
    if args.current_date:
        config['CURRENT_DATE'] = args.current_date
    if args.included_projects:
        config['INCLUDED_PROJECTS'] = [proj.upper() for proj in args.included_projects]

    print("=" * 80)
    print("INTEGRATED MINING CLAIM REDISTRIBUTION SYSTEM")
    print("=" * 80)
    print(f"Projects to include: {', '.join(config['INCLUDED_PROJECTS'])}")
    print("=" * 80)

    try:
        # Step 1: Process Midland
        midland_df = process_midland_file(config['MIDLAND_XLSX'], config['PROPERTY_CSV'], config)

        # Step 2: Process Wallbridge
        wallbridge_df = process_wallbridge_file(config['WALLBRIDGE_XLSX'], config['PROPERTY_CSV'], config)

        # Step 3: Merge datasets
        merged_csv = merge_datasets(midland_df, wallbridge_df, config)

        # Step 4-9: Run redistribution simulation
        gdf, gdf_outlines = load_and_prepare_data(merged_csv, config['SHP_PATH'], config['OUTLINES_SHP'], config)
        gdf = initialize_simulation(gdf)

        # === OPTIONAL: Add exploration program credits before simulation ===
        credits_added = False
        credited_titles = []  # Track title numbers that received credits for map highlighting
        if not args.skip_credits_prompt:
            print("\n" + "=" * 80)
            print("OPTIONAL: ADD EXPLORATION PROGRAM CREDITS")
            print("=" * 80)
            print("You can add credits from completed exploration programs before running")
            print("the simulation. Credits should be in an Excel file with columns:")
            print("  - Title Number (String): The claim title number")
            print("  - Amount (Float): The credit amount in dollars")
            print("  - Start Date (YYYY-MM-DD): When the credits were allocated")
            print("\nCredits will expire after 12 years (6 renewal terms × 2 years per Quebec Mining Act).")
            print("=" * 80)

            if args.credits_xlsx:
                # Use command-line provided path
                print(f"\nCredits file provided via command line: {args.credits_xlsx}")
                add_credits = prompt_yes_no("Do you want to load these credits?", default=True)
                if add_credits:
                    try:
                        gdf, num_credits, total_credits, credited_titles = load_and_apply_credits(args.credits_xlsx, gdf, config['OUTPUT_DIR'])
                        credits_added = True
                        print(f"\nSuccessfully added {num_credits} credit entries totaling ${total_credits:,.2f}")
                    except ValueError as e:
                        print(f"\nError loading credits: {e}")
                        if not prompt_yes_no("Continue simulation without these credits?", default=True):
                            print("Aborting simulation.")
                            return 1
            else:
                # Prompt user if they want to add credits
                add_credits = prompt_yes_no("\nDo you want to add credits from a completed exploration program?", default=False)

                if add_credits:
                    while True:
                        credits_path = input("Enter the path to the credits Excel file: ").strip()
                        if credits_path == '':
                            print("No path provided. Skipping credit addition.")
                            break

                        # Handle quoted paths
                        credits_path = credits_path.strip('"\'')

                        try:
                            gdf, num_credits, total_credits, credited_titles = load_and_apply_credits(credits_path, gdf, config['OUTPUT_DIR'])
                            credits_added = True
                            print(f"\nSuccessfully added {num_credits} credit entries totaling ${total_credits:,.2f}")
                            break
                        except ValueError as e:
                            print(f"\nError loading credits: {e}")
                            if not prompt_yes_no("Try a different file?", default=True):
                                print("Skipping credit addition.")
                                break
                        except FileNotFoundError:
                            print(f"\nFile not found: {credits_path}")
                            if not prompt_yes_no("Try a different file?", default=True):
                                print("Skipping credit addition.")
                                break
                        except Exception as e:
                            print(f"\nUnexpected error: {e}")
                            if not prompt_yes_no("Try a different file?", default=True):
                                print("Skipping credit addition.")
                                break

        # Store credits info for later reporting
        credits_info = {'added': credits_added, 'count': 0, 'total': 0.0}
        if credits_added:
            credits_info['count'] = num_credits
            credits_info['total'] = total_credits
            print("\n" + "=" * 80)
            print("Credits have been added. Proceeding with simulation...")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("No additional credits added. Proceeding with simulation...")
            print("=" * 80)

        dist_matrix, sindex = precompute_spatial_data(gdf)
        current_date = date.fromisoformat(config['CURRENT_DATE'])
        log_table, unresolved = run_simulation(gdf, dist_matrix, sindex, current_date, config)
        export_results(log_table, unresolved, gdf, config['OUTPUT_DIR'], current_date)
        plot_results(gdf, gdf_outlines, config['OUTPUT_DIR'], credited_titles)
        plot_interactive_map(gdf, log_table, unresolved, gdf_outlines, config['OUTPUT_DIR'])
        plot_summary_by_project(gdf, unresolved, config['OUTPUT_DIR'])

        # Step 10: Generate PDF report
        generate_pdf_report(gdf, log_table, config['OUTPUT_DIR'], current_date)

        # Print summary
        print("\n" + "=" * 80)
        print("SIMULATION COMPLETE")
        print("=" * 80)
        if credits_info['added']:
            print(f"Exploration credits added: {credits_info['count']} entries totaling ${credits_info['total']:,.2f}")
        print(f"Total claims analyzed: {len(gdf)}")
        print(f"Average years of life: {gdf['years_of_life'].mean():.2f}")
        print(f"Minimum years of life: {gdf['years_of_life'].min():.2f}")
        print(f"Maximum years of life: {gdf['years_of_life'].max():.2f}")
        print(f"Total credits redistributed: {sum(entry.get('credits_pulled', 0) for entry in log_table if entry['action_type'] == 'redistribution'):.2f}")
        print(f"Claims lapsed: {len(unresolved)}")
        print(f"\nResults saved to: {config['OUTPUT_DIR']}")
        if REPORTLAB_AVAILABLE:
            print(f"PDF report generated in output directory")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
