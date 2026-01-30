"""
This script simulates the renewal process for mining claims by performing recurring credit redistributions among nearby claims 
(within a configurable distance) to meet renewal requirements. The simulation processes claims chronologically based on their expiry dates, 
redistributing excess credits to cover deficits, prioritizing donors with earliest-expiring credits or distance/surplus scores. 
Credits can be transferred between claims on different properties, except for Casault claims, which are restricted to same-property transfers 
and cannot donate to non-Casault claims. Specified properties (DETOUR EAST, Radisson JV, NANTEL, HWY 810 - PUISSEAUX, HWY 810 - RAYMOND, 
N2, FENELON BM 864) are excluded. The loop continues until all claims are renewed up to a maximum number of renewals or lapse. 
Unresolved deficits are logged and summarized in a pivot table for the next 20 years. Dynamically updates the 'Expiration Dates and Amounts' 
column for redistributions and self-renewals, logging original and updated expiration logs. Visualizations include a static map with lapse years, 
an interactive map with toggleable redistribution flows, and a summary plot of claims by project and lapse year (with years ascending), 
all using a red-orange-yellow-green-blue gradient. Includes configuration file support, input validation, progress bars, and a summary report.
"""

# Import libraries for geospatial data handling, numerical operations, plotting, and file operations
import geopandas as gpd  # For handling geospatial data (GeoDataFrames)
import numpy as np  # For numerical computations (e.g., distance matrix)
import matplotlib.pyplot as plt  # For creating visualizations
import matplotlib.colors as mcolors  # For custom colormaps
from matplotlib.patches import Patch  # For creating legend patches in plots
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar  # For adding scale bars to plots
import pandas as pd  # For data manipulation (DataFrames)
import seaborn as sns  # For summary bar plot
import os  # For file and directory operations
from datetime import date  # For handling dates
from dateutil.relativedelta import relativedelta  # For date arithmetic (e.g., adding years)
from typing import Optional, List, Dict, Any, Tuple  # For type hints
import re  # For parsing strings with regular expressions
import copy  # For deep copying lists
import argparse  # For command-line arguments
import folium  # For interactive maps
from folium.plugins import Draw  # For interactive map drawing tools
import json  # For configuration file support
from tqdm import tqdm  # For progress bar

# Configuration dictionary for tunable parameters
CONFIG = {
    'MAX_DISTANCE': 3900.0,  # Maximum distance for credit redistribution (meters)
    'MAX_YEAR': 2060,  # Maximum year to prevent infinite simulation
    'MAX_RENEWALS': 6,  # Maximum renewals (simulation cap, not statutory)
    'SCORING_MODE': 'earliest_expiry',  # Scoring mode: 'earliest_expiry' or 'distance_surplus'
    'SCORING_WEIGHTS': {'surplus': 0.3, 'distance': 0.7},  # Weights for distance_surplus mode
    'CSV_PATH': r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Translated\Gestim_Wallbridge_Midland_20260109.csv",
    'SHP_PATH': r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Shapefile\gsm_claims_20250703.shp",
    'OUTLINES_SHP': r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Shapefile\wmc_property_outlines.shp",
    'OUTPUT_DIR': r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Redistribution",
    'CURRENT_DATE': '2026-01-09'  # Simulation start date
}

# Purpose: Load configuration from a JSON file or use defaults
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a JSON file, falling back to defaults.

    Args:
        config_path (Optional[str]): Path to JSON configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    config = CONFIG.copy()
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config.update(json.load(f))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_path}: {str(e)}")
    return config

# Purpose: Load and prepare data from CSV (attributes) and shapefiles (geometry, outlines) for the simulation
def load_and_prepare_data(csv_path: str, shp_path: str, outlines_path: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load and prepare data from CSV (attributes) and shapefiles (geometry for claims and outlines).

    Args:
        csv_path (str): Path to updated claims CSV file.
        shp_path (str): Path to claims shapefile for geometry.
        outlines_path (str): Path to outlines shapefile.

    Returns:
        Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: Prepared claims and outlines GeoDataFrames.
    """
    # Check if the CSV file exists at the specified path
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")  # Raise error if CSV is missing
    # Check if the claims shapefile exists
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Shapefile not found at: {shp_path}")  # Raise error if shapefile is missing
    # Check if the outlines shapefile exists
    if not os.path.exists(outlines_path):
        raise FileNotFoundError(f"Outlines shapefile not found at: {outlines_path}")  # Raise error if outlines shapefile is missing

    # Load claims shapefile into a GeoDataFrame for geometry data
    gdf_claims: gpd.GeoDataFrame = gpd.read_file(shp_path)  # Read shapefile with claim geometries

    # Load attributes from CSV into a pandas DataFrame
    df_csv = pd.read_csv(csv_path, encoding='utf-8')  # Read CSV with UTF-8 encoding for special characters

    # Validate required CSV columns
    required_cols = ['Title Number', 'Expiration Date', 'Surpluses', 'Required Works', 'Property', 'Area (Ha)', 'Number of Deadlines', 'Number of Renewals', 'Expiration Dates and Amounts']
    missing_cols = [col for col in required_cols if col not in df_csv.columns]
    if missing_cols:
        raise ValueError(f"Missing required CSV columns: {missing_cols}")  # Raise error if required columns are missing

    # Convert title_no to string and remove whitespace in both DataFrames for consistent merging
    gdf_claims['title_no'] = gdf_claims['title_no'].astype(str).str.strip()  # Convert shapefile title_no to string and strip spaces
    df_csv['Title Number'] = df_csv['Title Number'].astype(str).str.strip()  # Convert CSV Title Number to string and strip spaces

    # Validate title_no consistency
    csv_titles = set(df_csv['Title Number'])
    shp_titles = set(gdf_claims['title_no'])
    if not csv_titles.intersection(shp_titles):
        raise ValueError("No matching title_no values between CSV and shapefile")
    if len(csv_titles.intersection(shp_titles)) < len(csv_titles):
        print(f"Warning: {len(csv_titles) - len(csv_titles.intersection(shp_titles))} title_no values in CSV not found in shapefile")

    # Rename any project-like column in shapefile to avoid conflicts during merge
    for col in ['project', 'Property', 'PROJECT', 'property']:
        if col in gdf_claims.columns:
            gdf_claims = gdf_claims.rename(columns={col: f'{col}_shp'})  # Rename to avoid overwriting CSV's project column

    # Column mapping for CSV
    column_mapping: Dict[str, str] = {
        'Title Number': 'title_no',  # Map CSV Title Number to title_no
        'Expiration Date': 'expiry_date',  # Map Expiration Date to expiry_date
        'Surpluses': 'excess_work',  # Map Surpluses to excess_work
        'Required Works': 'required_work',  # Map Required Works to required_work
        'Property': 'project',  # Map Property to project
        'Area (Ha)': 'area_ha',  # Map Area (Ha) to area_ha
        'Number of Deadlines': 'terms_completed',  # Map Number of Deadlines to terms_completed
        'Number of Renewals': 'renewals_done',  # Map Number of Renewals to renewals_done
        'Expiration Dates and Amounts': 'credit_expirations_raw',  # Map expiration column for parsing
    }
    df_csv = df_csv.rename(columns=column_mapping)  # Apply column renaming to CSV DataFrame

    # Convert numeric columns to float, handling commas and missing values
    for col in ['excess_work', 'required_work', 'area_ha']:
        df_csv[col] = (
            df_csv[col]
            .astype(str)  # Convert to string to handle various input types
            .str.replace(',', '.')  # Replace comma with dot for decimal numbers
            .replace(['None', ''], '0')  # Replace None or empty with 0
            .astype(float)  # Convert to float
            .round(2)  # Round to 2 decimal places for precision
        )
    # Convert integer columns, filling missing values with 0
    for col in ['terms_completed', 'renewals_done']:
        df_csv[col] = df_csv[col].fillna(0).astype(int)  # Fill NaN with 0 and convert to int

    # Define function to parse Expiration Dates and Amounts into a sorted list of date-amount dictionaries
    def parse_expirations(exp_str: str) -> List[Dict[str, Any]]:
        """
        Parse expiration string into a sorted list of {'date': Timestamp, 'amount': float}.

        Example:
            Input: "2028/10/16 (1377,81 $); 2038/10/16 (636,54 $)"
            Output: [{'date': Timestamp('2028-10-16'), 'amount': 1377.81}, {'date': Timestamp('2038-10-16'), 'amount': 636.54}]
        """
        if pd.isna(exp_str) or exp_str.strip() == '':  # Check if string is NaN or empty
            return []  # Return empty list if no data
        # Split by semicolon to separate multiple expiration entries
        entries = exp_str.split(';')
        expirations = []
        for entry in entries:
            entry = entry.strip()  # Remove leading/trailing whitespace
            if not entry:  # Skip empty entries
                continue
            # Match format: YYYY/MM/DD (AMOUNT $)
            match = re.match(r'(\d{4}/\d{2}/\d{2})\s*\(([\d,.]+)\s*\$?\)', entry)
            if match:
                date_str, amount_str = match.groups()  # Extract date and amount
                try:
                    exp_date = pd.to_datetime(date_str, format='%Y/%m/%d')  # Parse date as Timestamp
                    amount = round(float(amount_str.replace(',', '.')), 2)  # Convert amount to float, round to 2 decimals
                    if amount > 0:  # Skip zero-amount entries
                        expirations.append({'date': exp_date, 'amount': amount})  # Add to list
                except (ValueError, TypeError):
                    print(f"Warning: Could not parse expiration entry: {entry}")  # Log parsing errors
            else:
                print(f"Warning: Invalid expiration format: {entry}")  # Log invalid format
        # Sort expirations by date ascending
        expirations.sort(key=lambda x: x['date'])
        return expirations

    # Apply parsing to create credit_expirations column
    df_csv['credit_expirations'] = df_csv['credit_expirations_raw'].apply(parse_expirations)  # Parse expiration data

    # Validate that sum of expiration amounts matches excess_work
    for idx, row in df_csv.iterrows():
        exp_sum = sum(exp['amount'] for exp in row['credit_expirations'])  # Sum expiration amounts
        if abs(exp_sum - row['excess_work']) > 0.01:  # Allow small floating-point tolerance
            print(f"Warning: For title_no {row['title_no']}, expiration sum {exp_sum} does not match excess_work {row['excess_work']}")  # Log mismatch

    # Merge CSV attributes into shapefile GeoDataFrame on title_no
    gdf = gdf_claims.merge(
        df_csv[['title_no', 'expiry_date', 'excess_work', 'required_work', 'project', 'area_ha', 'terms_completed', 'renewals_done', 'credit_expirations']],
        on='title_no',  # Merge on title_no
        how='inner',  # Keep only matching records
        suffixes=('_shp', None)  # Add _shp suffix to shapefile columns if conflicts occur
    )

    # Check if merge is empty
    if gdf.empty:
        raise ValueError("Merge resulted in an empty DataFrame. Check title_no matches between CSV and shapefile.")  # Raise error if merge is empty

    # Ensure project column exists, fallback to shapefile if needed
    if 'project' not in gdf.columns:
        for col in ['project_shp', 'Property_shp', 'PROJECT_shp', 'property_shp']:
            if col in gdf.columns:
                gdf['project'] = gdf[col]  # Use shapefile column as fallback
                print(f"Warning: Using {col} as project column from shapefile.")  # Log fallback usage
                break
        else:
            raise KeyError("Column 'project' not found in merged GeoDataFrame and no fallback project-like column found in shapefile.")  # Raise error if no project column

    # Ensure CRS is EPSG:2958 (UTM Zone 17N) for spatial consistency
    if gdf.crs != "EPSG:2958":
        gdf = gdf.to_crs(epsg=2958)  # Convert to UTM Zone 17N

    # Filter out specified projects (case-sensitive)
    gdf = gdf[~gdf['project'].isin(['DETOUR EAST', 'Radisson JV', 'NANTEL', 'HWY 810 - PUISSEAUX', 'HWY 810 - RAYMOND', 'N2', 'FENELON BM 864'])].copy().reset_index(drop=True)  # Exclude specified projects and reset index

    # Load outlines shapefile for property boundaries
    gdf_outlines: gpd.GeoDataFrame = gpd.read_file(outlines_path)  # Read outlines shapefile
    if gdf_outlines.crs != "EPSG:2958":
        gdf_outlines = gdf_outlines.to_crs(epsg=2958)  # Convert outlines to UTM Zone 17N

    return gdf, gdf_outlines  # Return prepared claims and outlines GeoDataFrames

# Purpose: Initialize the GeoDataFrame with simulation-specific columns
def initialize_simulation(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Initialize columns for the simulation.

    Args:
        gdf (gpd.GeoDataFrame): Prepared claims GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: Initialized GeoDataFrame.
    """
    # Copy excess_work to track available credits during simulation
    gdf['current_excess_work'] = gdf['excess_work']  # Initialize current excess credits
    # Copy required_work for simulation
    gdf['current_required_work'] = gdf['required_work']  # Initialize current required credits
    # Convert expiry_date to datetime, handling errors
    gdf['current_expiry_date'] = pd.to_datetime(gdf['expiry_date'], errors='coerce')  # Parse expiry dates
    # Drop rows with invalid expiry dates
    gdf = gdf.dropna(subset=['current_expiry_date'])  # Remove claims with missing expiry dates
    # Mark all claims as active initially
    gdf['active'] = True  # Set active status to True
    # Initialize renewal counter
    gdf['renewals'] = 0  # Set initial renewals to 0
    # Calculate current term based on completed terms and renewals
    gdf['current_term'] = gdf['terms_completed'] + gdf['renewals_done'] + 1  # Compute current term number
    # Copy expiry date for final tracking
    gdf['final_expiry_date'] = gdf['current_expiry_date'].copy()  # Initialize final expiry date
    # Compute centroids for spatial calculations
    gdf['centroid'] = gdf.geometry.centroid  # Calculate geometric centroids

    # Convert to EPSG:4326 (WGS84) to compute latitude in degrees
    gdf_4326 = gdf.to_crs(epsg=4326)  # Transform to WGS84 for latitude
    # Extract latitude from centroids
    gdf['latitude'] = gdf_4326['centroid'].y  # Assign latitude values

    return gdf  # Return initialized GeoDataFrame

# Purpose: Calculate required work credits based on Quebec mining regulations
def get_required_work(term: int, area_ha: float, latitude: float) -> float:
    """
    Compute required minimum work cost based on Quebec regulation.

    Args:
        term (int): Current term number.
        area_ha (float): Area in hectares.
        latitude (float): Latitude for north/south determination.

    Returns:
        float: Required minimum cost.
    """
    # Determine if claim is north of 52°N
    is_north = latitude > 52  # Check if latitude is above 52°N
    # Define cost tables based on location and area
    if is_north:
        if area_ha < 25:  # Small claims
            costs = [48, 160, 320, 480, 640, 750, 1000]  # Costs for terms 1-7+
        elif area_ha <= 45:  # Medium claims
            costs = [120, 400, 800, 1200, 1600, 1800, 2500]  # Costs for terms 1-7+
        else:  # Large claims
            costs = [135, 450, 900, 1350, 1800, 1800, 2500]  # Costs for terms 1-7+
    else:
        if area_ha < 25:  # Small claims
            costs = [500, 500, 500, 750, 750, 750, 1200]  # Costs for terms 1-7+
        elif area_ha <= 100:  # Medium claims
            costs = [1200, 1200, 1200, 1800, 1800, 1800, 2500]  # Costs for terms 1-7+
        else:  # Large claims
            costs = [1800, 1800, 1800, 2700, 2700, 2700, 3600]  # Costs for terms 1-7+
    # Cap term index at 6 (for terms 7+)
    idx = min(term - 1, 6)  # Get index for cost table
    return costs[idx]  # Return required work cost

# Purpose: Precompute spatial data for efficient distance calculations
def precompute_spatial_data(gdf: gpd.GeoDataFrame) -> Tuple[np.ndarray, Any]:
    """
    Precompute distance matrix and spatial index.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with centroids.

    Returns:
        Tuple[np.ndarray, Any]: Distance matrix and spatial index.
    """
    # Extract centroid coordinates as NumPy array
    points = np.array([list(p.coords[0]) for p in gdf.centroid])  # Convert centroids to array of [x, y]
    # Compute pairwise Euclidean distances
    dist_matrix = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=-1)  # Calculate distance matrix

    # Create GeoDataFrame with centroids for spatial index
    cent_gdf = gpd.GeoDataFrame(geometry=gdf.centroid, crs=gdf.crs)  # Create centroid GeoDataFrame
    cent_gdf.index = gdf.index  # Set same index as input
    # Build spatial index for efficient querying
    sindex = cent_gdf.sindex  # Create spatial index

    return dist_matrix, sindex  # Return distance matrix and spatial index

# Purpose: Reduce credit amounts from the earliest expiration entries and return the pulled portions
def reduce_expirations(exp_list: List[Dict[str, Any]], amount: float) -> List[Dict[str, Any]]:
    """
    Reduce the specified amount from the earliest expirations, returning the pulled portions.

    Args:
        exp_list (List[Dict[str, Any]]): List of {'date': Timestamp, 'amount': float}, sorted by date.
        amount (float): Amount to reduce.

    Returns:
        List[Dict[str, Any]]: List of pulled portions {'date': Timestamp, 'amount': float}, excluding zero amounts.

    Example:
        Input: exp_list=[{'date': Timestamp('2028-10-16'), 'amount': 1377.81}, {'date': Timestamp('2038-10-16'), 'amount': 636.54}], amount=1000
        Output: [{'date': Timestamp('2028-10-16'), 'amount': 1000}]
        Modifies exp_list to [{'date': Timestamp('2028-10-16'), 'amount': 377.81}, {'date': Timestamp('2038-10-16'), 'amount': 636.54}]
    """
    pulled_portions = []  # Initialize list to store pulled portions
    used = 0  # Track total amount used
    i = 0  # Initialize index
    while used < amount and i < len(exp_list):  # Continue until amount is met or list is exhausted
        pull = min(amount - used, exp_list[i]['amount'])  # Calculate amount to pull from current entry
        if pull > 0:  # Only include non-zero pulls
            exp_list[i]['amount'] -= pull  # Reduce entry amount
            pulled_portions.append({'date': exp_list[i]['date'], 'amount': pull})  # Record pulled portion
            used += pull  # Update total used
        if exp_list[i]['amount'] <= 0:  # If entry is depleted
            del exp_list[i]  # Remove entry
        else:
            i += 1  # Move to next entry
    return pulled_portions  # Return list of pulled portions

# Purpose: Format expiration list as a readable string
def format_expirations(exp_list: List[Dict[str, Any]]) -> str:
    """
    Format expiration list as a semicolon-separated string.

    Example:
        Input: [{'date': Timestamp('2028-10-16'), 'amount': 1377.81}, {'date': Timestamp('2038-10-16'), 'amount': 636.54}]
        Output: "2028/10/16 (1377.81 $); 2038/10/16 (636.54 $)"
    """
    if not isinstance(exp_list, list):  # If already a string, return it
        return exp_list
    return '; '.join(f"{exp['date'].strftime('%Y/%m/%d')} ({exp['amount']:.2f} $)" for exp in exp_list if exp['amount'] > 0) or ''  # Format as "YYYY/MM/DD (AMOUNT $); ...", exclude zero amounts

# Purpose: Run the core simulation, handling renewals, credit redistributions, and expirations
def run_simulation(gdf: gpd.GeoDataFrame, dist_matrix: np.ndarray, sindex: Any, current_date: date, config: Dict[str, Any] = CONFIG) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run the main simulation loop for claim renewals and redistributions, allowing credit transfers between different properties
    except for Casault claims, which are restricted to same-property transfers and cannot donate to non-Casault claims, and dynamically updating credit expiration logs.

    Args:
        gdf (gpd.GeoDataFrame): Initialized GeoDataFrame with claim data.
        dist_matrix (np.ndarray): Precomputed distance matrix for claims.
        sindex (Any): Spatial index for efficient querying.
        current_date (date): Current simulation date.
        config (Dict[str, Any]): Configuration dictionary with parameters like MAX_DISTANCE, SCORING_MODE.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Log table of redistributions and unresolved deficits list.
    """
    MAX_DISTANCE: float = config['MAX_DISTANCE']  # Maximum distance for credit redistribution
    MAX_YEAR: int = config['MAX_YEAR']  # Maximum year to prevent infinite simulation
    MAX_RENEWALS: int = config['MAX_RENEWALS']  # Maximum renewals
    SCORING_MODE: str = config['SCORING_MODE']  # Scoring mode for donor selection
    w1 = config['SCORING_WEIGHTS']['surplus']  # Weight for surplus
    w2 = config['SCORING_WEIGHTS']['distance']  # Weight for distance
    log_table: List[Dict[str, Any]] = []  # Initialize list to log credit transfers and renewals
    unresolved: List[Dict[str, Any]] = []  # Initialize list to log unresolved deficits

    # Main simulation loop
    while True:
        # Filter active claims
        active_claims = gdf[gdf['active']]  # Get currently active claims
        # Exit loop if no active claims remain
        if active_claims.empty:
            break  # Stop simulation if no claims are active
        # Find earliest expiry date among active claims
        next_date = active_claims['current_expiry_date'].min()  # Get next expiry date
        # Check for invalid date or year limit
        if pd.isna(next_date) or next_date.year > MAX_YEAR:
            gdf.at[active_claims.index, 'active'] = False  # Deactivate all claims
            break  # Exit if date is invalid or beyond max year

        # Apply credit expirations on or before next_date for all active claims
        for idx in tqdm(active_claims.index, desc=f"Processing expirations for {next_date}"):
            exp_list = gdf.at[idx, 'credit_expirations']  # Get credit expiration data
            to_remove = 0  # Initialize total expired credits
            new_list = []  # Initialize new expiration list
            for exp in exp_list:
                if exp['date'] > next_date:  # Keep future expirations
                    new_list.append(exp)
                else:
                    to_remove += exp['amount']  # Sum expired amounts
            gdf.at[idx, 'credit_expirations'] = new_list  # Update expiration list
            gdf.at[idx, 'current_excess_work'] -= to_remove  # Reduce excess work
            # Verify consistency between credit_expirations and current_excess_work
            if abs(sum(exp['amount'] for exp in new_list) - gdf.at[idx, 'current_excess_work']) > 0.01:
                print(f"Warning: For title_no {gdf.at[idx, 'title_no']}, expiration sum does not match current_excess_work after expiration")

        # Get claims expiring on next_date
        expiring = active_claims[active_claims['current_expiry_date'] == next_date]  # Filter claims expiring now
        # Calculate deficits (required - excess work)
        expiring = expiring.assign(deficit=np.maximum(0, expiring['current_required_work'] - expiring['current_excess_work']))  # Compute deficits
        # Sort by deficit ascending to prioritize low-deficit claims
        expiring = expiring.sort_values(by='deficit', ascending=True)  # Sort for processing order

        # Process each expiring claim
        for idx, row in tqdm(expiring.iterrows(), desc=f"Processing claims expiring on {next_date}", total=len(expiring)):
            # Check if claim has reached maximum renewals
            if row['renewals'] >= MAX_RENEWALS:
                # Log unresolved deficit
                unresolved.append({
                    'year': next_date.year,  # Year of lapse
                    'project': row['project'],  # Project name
                    'deficit_needed': row['current_required_work'] - row['current_excess_work'],  # Deficit amount
                    'title_no': row['title_no']  # Claim ID
                })
                gdf.at[idx, 'active'] = False  # Deactivate claim
                continue  # Skip to next claim
            # Get current excess and required work
            excess = gdf.at[idx, 'current_excess_work']  # Current available credits
            required = gdf.at[idx, 'current_required_work']  # Required credits for renewal
            deficit = required - excess  # Calculate deficit
            # Check if claim can self-renew
            if deficit <= 0:
                # Copy original expirations for logging
                original_exp = copy.deepcopy(gdf.at[idx, 'credit_expirations'])  # Copy original expiration list
                # Self-renew: Reduce from earliest expirations
                pulled_portions = reduce_expirations(gdf.at[idx, 'credit_expirations'], required)  # Reduce required amount from expirations
                gdf.at[idx, 'current_excess_work'] -= required  # Deduct required credits
                gdf.at[idx, 'current_expiry_date'] += relativedelta(years=2)  # Extend expiry by 2 years
                gdf.at[idx, 'renewals'] += 1  # Increment renewal count
                gdf.at[idx, 'current_term'] += 1  # Increment term number
                # Update required work for next term
                gdf.at[idx, 'current_required_work'] = get_required_work(gdf.at[idx, 'current_term'], row['area_ha'], row['latitude'])
                gdf.at[idx, 'final_expiry_date'] = gdf.at[idx, 'current_expiry_date']  # Update final expiry
                # Log the renewal
                log_entry = {
                    'action_type': 'renewal',  # Type of action
                    'title_no': row['title_no'],  # Claim ID
                    'renewal_date': next_date,  # Date of renewal
                    'renewal_year': next_date.year,  # Year of renewal
                    'renewal_amount': required,  # Amount paid for renewal
                    'pulled_expirations': format_expirations(pulled_portions),  # Formatted string of expirations used for renewal
                    'original_expirations': format_expirations(original_exp),  # Original expiration log
                    'updated_expirations': format_expirations(gdf.at[idx, 'credit_expirations'])  # Updated expiration log
                }
                log_table.append(log_entry)  # Add to log table
                # Verify consistency
                if abs(sum(exp['amount'] for exp in gdf.at[idx, 'credit_expirations']) - gdf.at[idx, 'current_excess_work']) > 0.01:
                    print(f"Warning: For title_no {row['title_no']}, expiration sum does not match current_excess_work after renewal")
                continue  # Move to next claim
            # Calculate credits needed to cover deficit
            credits_needed = deficit  # Amount of credits to source
            # Create buffer for spatial query
            buffer_geo = row['centroid'].buffer(MAX_DISTANCE)  # Buffer around centroid for nearby claims
            # Query nearby claims using spatial index
            candidates_pos = sindex.query(buffer_geo, predicate='intersects')  # Find nearby claim indices
            nearby = gdf.iloc[candidates_pos]  # Get nearby claims
            # Filter to active claims with surplus, excluding self
            nearby = nearby[
                (nearby['active']) & 
                (nearby.index != idx) & 
                (nearby['current_excess_work'] > nearby['current_required_work'])
            ]
            # Apply Casault isolation: Casault claims only interact with Casault claims (neither receive nor donate outside their property)
            if row['project'] == 'CASAULT':
                nearby = nearby[nearby['project'] == 'CASAULT']  # Casault recipients only from Casault (same property)
            else:
                nearby = nearby[nearby['project'] != 'CASAULT']  # Non-Casault recipients only from non-Casault (free transfer within distance)
            # Process nearby donors if any exist
            if not nearby.empty:
                # Assign distance and available credits
                nearby = nearby.assign(
                    distance=dist_matrix[idx][nearby.index],  # Get distances from precomputed matrix
                    available=nearby['current_excess_work'] - nearby['current_required_work']  # Calculate available surplus
                )
                # Compute donor scores based on scoring mode
                if SCORING_MODE == 'earliest_expiry':
                    # Prioritize donors with earliest-expiring credits
                    nearby = nearby.assign(
                        score=[min((exp['date'] for exp in gdf.at[idx, 'credit_expirations']), default=pd.Timestamp.max).timestamp() for idx in nearby.index]
                    )
                    nearby = nearby.sort_values(by='score', ascending=True)  # Earliest expiry first
                else:  # distance_surplus mode
                    # Get min/max for normalization
                    min_avail = nearby['available'].min()  # Minimum available credits
                    max_avail = nearby['available'].max()  # Maximum available credits
                    min_dist = nearby['distance'].min()  # Minimum distance
                    max_dist = nearby['distance'].max()  # Maximum distance
                    # Normalize surplus
                    if max_avail > min_avail:
                        norm_surplus = (nearby['available'] - min_avail) / (max_avail - min_avail)  # Normalize available credits
                    else:
                        norm_surplus = pd.Series(1.0, index=nearby.index)  # Default to 1 if no range
                    # Normalize distance using logarithmic scaling
                    if max_dist > min_dist:
                        log_dist = np.log1p(nearby['distance'])  # Apply log1p to compress distance range
                        log_min = np.log1p(min_dist)  # Log of min distance
                        log_max = np.log1p(max_dist)  # Log of max distance
                        norm_distance = (log_dist - log_min) / (log_max - log_min)  # Normalize distance
                    else:
                        norm_distance = pd.Series(0.0, index=nearby.index)  # Default to 0 if no range
                    # Compute score
                    nearby = nearby.assign(score=(w1 * norm_surplus) + (w2 * (1 - norm_distance)))  # Combine weighted scores
                    nearby = nearby.sort_values(by='score', ascending=False)  # Prioritize high-scoring donors
                
                # Process each donor
                for donor_idx, donor in nearby.iterrows():
                    # Calculate available credits for donor
                    avail = gdf.at[donor_idx, 'current_excess_work'] - gdf.at[donor_idx, 'current_required_work']  # Donor’s surplus
                    pull = min(avail, credits_needed)  # Amount to pull (min of available and needed)
                    if pull > 0:
                        # Copy original donor expirations for logging
                        original_exp_donor = copy.deepcopy(gdf.at[donor_idx, 'credit_expirations'])  # Copy original donor expiration list
                        # Reduce from donor's earliest expirations and get pulled portions
                        pulled_portions = reduce_expirations(gdf.at[donor_idx, 'credit_expirations'], pull)  # Reduce donor’s expirations
                        gdf.at[donor_idx, 'current_excess_work'] -= pull  # Deduct from donor
                        # Add pulled portions to recipient’s expirations
                        gdf.at[idx, 'credit_expirations'].extend(pulled_portions)  # Append donor’s expiration portions
                        gdf.at[idx, 'credit_expirations'] = sorted(gdf.at[idx, 'credit_expirations'], key=lambda d: d['date'])  # Sort recipient’s expirations
                        gdf.at[idx, 'current_excess_work'] += pull  # Add to recipient
                        # Verify consistency
                        if abs(sum(exp['amount'] for exp in gdf.at[idx, 'credit_expirations']) - gdf.at[idx, 'current_excess_work']) > 0.01:
                            print(f"Warning: For title_no {row['title_no']}, expiration sum does not match current_excess_work after redistribution")
                        # Log the transfer with donor original and updated expirations
                        log_entry = {
                            'action_type': 'redistribution',  # Type of action
                            'insufficient_title_no': row['title_no'],  # Recipient claim ID
                            'recipient_project': row['project'],  # Recipient project
                            'expiry_date': next_date,  # Date of transfer
                            'redistribution_year': next_date.year,  # Year of redistribution
                            'deficit_needed': pull,  # Amount needed
                            'donor_title_no': donor['title_no'],  # Donor claim ID
                            'donor_project': donor['project'],  # Donor project
                            'donor_original_excess': donor['current_excess_work'],  # Donor’s original excess
                            'donor_expiration_date': donor['current_expiry_date'],  # Donor’s expiry date
                            'credits_pulled': pull,  # Amount transferred
                            'donor_new_excess_work': gdf.at[donor_idx, 'current_excess_work'],  # Donor’s new excess
                            'pulled_expirations': format_expirations(pulled_portions),  # Formatted string of expirations pulled from donor
                            'original_expirations_donor': format_expirations(original_exp_donor),  # Original donor expiration log
                            'updated_expirations_donor': format_expirations(gdf.at[donor_idx, 'credit_expirations']),  # Updated donor expiration log
                            'distance_m': donor['distance']  # Distance between claims
                        }
                        log_table.append(log_entry)  # Add to log table
                        credits_needed -= pull  # Reduce remaining need
                    if credits_needed <= 0:
                        break  # Stop if deficit is covered
            # Check if deficit was resolved
            if credits_needed <= 0:
                # Renew: Reduce from earliest expirations
                original_exp = copy.deepcopy(gdf.at[idx, 'credit_expirations'])  # Copy original expiration list
                pulled_portions = reduce_expirations(gdf.at[idx, 'credit_expirations'], gdf.at[idx, 'current_required_work'])  # Reduce required amount from expirations
                gdf.at[idx, 'current_excess_work'] -= gdf.at[idx, 'current_required_work']  # Deduct required credits
                gdf.at[idx, 'current_expiry_date'] += relativedelta(years=2)  # Extend expiry by 2 years
                gdf.at[idx, 'renewals'] += 1  # Increment renewal count
                gdf.at[idx, 'current_term'] += 1  # Increment term
                # Update required work for next term
                gdf.at[idx, 'current_required_work'] = get_required_work(gdf.at[idx, 'current_term'], row['area_ha'], row['latitude'])
                gdf.at[idx, 'final_expiry_date'] = gdf.at[idx, 'current_expiry_date']  # Update final expiry
                # Log the renewal
                log_entry = {
                    'action_type': 'renewal',  # Type of action
                    'title_no': row['title_no'],  # Claim ID
                    'renewal_date': next_date,  # Date of renewal
                    'renewal_year': next_date.year,  # Year of renewal
                    'renewal_amount': gdf.at[idx, 'current_required_work'],  # Amount paid for renewal
                    'pulled_expirations': format_expirations(pulled_portions),  # Formatted string of expirations used for renewal
                    'original_expirations': format_expirations(original_exp),  # Original expiration log
                    'updated_expirations': format_expirations(gdf.at[idx, 'credit_expirations'])  # Updated expiration log
                }
                log_table.append(log_entry)  # Add to log table
                # Verify consistency
                if abs(sum(exp['amount'] for exp in gdf.at[idx, 'credit_expirations']) - gdf.at[idx, 'current_excess_work']) > 0.01:
                    print(f"Warning: For title_no {row['title_no']}, expiration sum does not match current_excess_work after renewal")
            else:
                # Log unresolved deficit
                unresolved.append({
                    'year': next_date.year,  # Year of lapse
                    'project': row['project'],  # Project name
                    'deficit_needed': credits_needed,  # Deficit amount
                    'title_no': row['title_no']  # Claim ID
                })
                gdf.at[idx, 'active'] = False  # Deactivate claim

    return log_table, unresolved  # Return transfer log and unresolved deficits

# Purpose: Export simulation results to CSV files
def export_results(log_table: List[Dict[str, Any]], unresolved: List[Dict[str, Any]], gdf: gpd.GeoDataFrame, output_dir: str, current_date: date) -> None:
    """
    Export simulation results to CSV files.

    Args:
        log_table (List[Dict[str, Any]]): Redistribution log.
        unresolved (List[Dict[str, Any]]): Unresolved deficits.
        gdf (gpd.GeoDataFrame): Final GeoDataFrame.
        output_dir (str): Output directory path.
        current_date (date): Current simulation date.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

    # Export redistribution log
    try:
        pd.DataFrame(log_table).to_csv(os.path.join(output_dir, 'full_redistribution_log.csv'), index=False)  # Save log table to CSV
    except PermissionError as e:
        print(f"PermissionError: Unable to write to {os.path.join(output_dir, 'full_redistribution_log.csv')}. {str(e)}")  # Log permission error
        print("Please ensure the directory is writable and no files are open in other applications.")  # Suggest fix
        raise  # Re-raise to halt execution

    # Process unresolved deficits if any exist
    if unresolved:
        try:
            unresolved_df = pd.DataFrame(unresolved)  # Convert unresolved list to DataFrame
            # Group by year and project, summing deficits
            grouped = unresolved_df.groupby(['year', 'project'])['deficit_needed'].sum().reset_index(name='total_money_needed')
            grouped.to_csv(os.path.join(output_dir, 'unresolved_credits_by_year_property.csv'), index=False)  # Save grouped deficits

            current_year = current_date.year  # Get current year
            max_year = current_year + 20  # Set 20-year horizon
            grouped_filtered = grouped[grouped['year'] <= max_year]  # Filter to 20 years
            if not grouped_filtered.empty:
                # Create pivot table of deficits by year and project
                pivot = grouped_filtered.pivot(index='year', columns='project', values='total_money_needed').fillna(0)
                pivot['Total'] = pivot.sum(axis=1)  # Add total column
                pivot.to_csv(os.path.join(output_dir, 'required_spend_pivot.csv'))  # Save pivot table
        except PermissionError as e:
            print(f"PermissionError: Unable to write unresolved credits files. {str(e)}")  # Log permission error
            print("Please ensure the directory is writable and no files are open in other applications.")  # Suggest fix
            raise  # Re-raise to halt execution

    # Calculate and export claim life expectancies
    try:
        gdf['days_of_life'] = (gdf['final_expiry_date'] - pd.Timestamp(current_date)).dt.days  # Compute days of life
        gdf['years_of_life'] = gdf['days_of_life'] / 365.25  # Convert to years
        # Save selected columns to CSV
        gdf[['title_no', 'project', 'expiry_date', 'final_expiry_date', 'renewals', 'years_of_life']].to_csv(
            os.path.join(output_dir, 'claims_years_of_life.csv'), index=False
        )
    except PermissionError as e:
        print(f"PermissionError: Unable to write to {os.path.join(output_dir, 'claims_years_of_life.csv')}. {str(e)}")  # Log permission error
        print("Please ensure the directory is writable and no files are open in other applications.")  # Suggest fix
        raise  # Re-raise to halt execution

    # Export expiration history
    try:
        expiration_history = []
        for idx, row in gdf.iterrows():
            expiration_history.append({
                'title_no': row['title_no'],  # Claim ID
                'project': row['project'],  # Project name
                'final_expiry_date': row['final_expiry_date'],  # Final expiry date
                'credit_expirations': format_expirations(row['credit_expirations'])  # Formatted expiration log
            })
        pd.DataFrame(expiration_history).to_csv(os.path.join(output_dir, 'claim_expiration_history.csv'), index=False)
    except PermissionError as e:
        print(f"PermissionError: Unable to write to {os.path.join(output_dir, 'claim_expiration_history.csv')}. {str(e)}")  # Log permission error
        print("Please ensure the directory is writable and no files are open in other applications.")  # Suggest fix
        raise

    # Generate summary report
    try:
        report = []
        report.append(f"Simulation Summary Report")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Claims Analyzed: {len(gdf)}")
        report.append(f"Average Years of Life: {gdf['years_of_life'].mean():.2f}")
        report.append(f"Minimum Years of Life: {gdf['years_of_life'].min():.2f}")
        report.append(f"Maximum Years of Life: {gdf['years_of_life'].max():.2f}")
        report.append(f"Total Credits Redistributed: {sum(entry['credits_pulled'] for entry in log_table if entry['action_type'] == 'redistribution'):.2f}")
        report.append(f"Claims Lapsed: {len(unresolved)}")
        with open(os.path.join(output_dir, 'simulation_report.txt'), 'w') as f:
            f.write('\n'.join(report))
        print(f"Exported summary report to {os.path.join(output_dir, 'simulation_report.txt')}")
    except PermissionError as e:
        print(f"PermissionError: Unable to write to {os.path.join(output_dir, 'simulation_report.txt')}. {str(e)}")
        print("Please ensure the directory is writable and no files are open in other applications.")
        raise

# Purpose: Generate and save an interactive map of claim life categories
def plot_interactive_map(gdf: gpd.GeoDataFrame, log_table: List[Dict[str, Any]], unresolved: List[Dict[str, Any]], gdf_outlines: gpd.GeoDataFrame, output_dir: str) -> None:
    """
    Create an interactive map using folium with claim lapse years and toggleable redistribution flows.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with final_expiry_date and credit_expirations.
        log_table (List[Dict[str, Any]]): Redistribution log with donor-recipient pairs.
        unresolved (List[Dict[str, Any]]): Unresolved deficits list.
        gdf_outlines (gpd.GeoDataFrame): Outlines GeoDataFrame.
        output_dir (str): Output directory path.
    """
    # Convert to WGS84 (EPSG:4326) for folium
    gdf_4326 = gdf.to_crs(epsg=4326)
    gdf_outlines_4326 = gdf_outlines.to_crs(epsg=4326)

    # Use pre-computed centroids from gdf (in EPSG:2958) and convert to WGS84
    gdf_centroids_4326 = gpd.GeoDataFrame(geometry=gdf['centroid'], crs='EPSG:2958').to_crs(epsg=4326)

    # Create base map centered on claims
    m = folium.Map(
        location=[gdf_centroids_4326.geometry.y.mean(), gdf_centroids_4326.geometry.x.mean()],
        zoom_start=10,
        tiles='Stamen Terrain',
        attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL'
    )

    # Add draw plugin for interactivity
    Draw(export=True).add_to(m)

    # Define color mapping for lapse years (red-orange-yellow-green-blue)
    unique_years = sorted(gdf['life_category'].unique())
    n_years = len(unique_years)
    if n_years > 0:
        colors = mcolors.LinearSegmentedColormap.from_list(
            'hot_to_cold', ['red', 'orange', 'yellow', 'green', 'blue'], N=n_years
        )(np.linspace(0, 1, n_years))
        cat_colors = {year: mcolors.to_hex(colors[i]) for i, year in enumerate(unique_years)}
    else:
        cat_colors = {}

    # Add claims with tooltips
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

    # Add property outlines
    outlines_group = folium.FeatureGroup(name='Property Outlines', show=True)
    folium.GeoJson(
        gdf_outlines_4326,
        style_function=lambda x: {'color': 'black', 'weight': 2, 'fillOpacity': 0}
    ).add_to(outlines_group)
    outlines_group.add_to(m)

    # Add redistribution flows as toggleable layer
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
            else:
                print(f"Warning: Skipping redistribution flow for donor {entry['donor_title_no']} or recipient {entry['insufficient_title_no']} not found in gdf_4326")
    flow_group.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add legend for lapse years
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

    # Save interactive map
    try:
        m.save(os.path.join(output_dir, 'interactive_gsm_claims_map.html'))
        print(f"Exported interactive map to '{os.path.join(output_dir, 'interactive_gsm_claims_map.html')}'")
    except PermissionError as e:
        print(f"PermissionError: Unable to write to {os.path.join(output_dir, 'interactive_gsm_claims_map.html')}. {str(e)}")
        print("Please ensure the directory is writable and no files are open in other applications.")
        raise

# Purpose: Generate and save a visualization of claim life categories by lapse year
def plot_results(gdf: gpd.GeoDataFrame, gdf_outlines: gpd.GeoDataFrame, output_dir: str) -> None:
    """
    Generate and save visualization of claim life categories, using specific lapse years with a bright red to orange to yellow to green to blue color gradient.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with final_expiry_date.
        gdf_outlines (gpd.GeoDataFrame): Outlines GeoDataFrame.
        output_dir (str): Output directory path.
    """
    # Define function to categorize claims by lapse year
    def get_life_category(final_expiry_date: pd.Timestamp) -> str:
        return str(final_expiry_date.year)  # Return year as string (e.g., '2032')

    # Apply lapse year category to each claim
    gdf['life_category'] = gdf['final_expiry_date'].apply(get_life_category)  # Assign lapse year categories

    # Define color mapping for lapse years (bright red to orange to yellow to green to blue)
    unique_years = sorted(gdf['life_category'].unique())  # Get unique lapse years in ascending order
    n_years = len(unique_years)  # Number of unique years
    if n_years > 0:
        # Create a custom hot-to-cold colormap
        colors = mcolors.LinearSegmentedColormap.from_list(
            'hot_to_cold', ['red', 'orange', 'yellow', 'green', 'blue'], N=n_years
        )(np.linspace(0, 1, n_years))  # Blend from bright red (early) to blue (late)
        cat_colors = {year: colors[i] for i, year in enumerate(unique_years)}  # Map years to colors
    else:
        cat_colors = {}  # Empty dict if no years

    # Create figure and axes for plotting
    fig, ax = plt.subplots(figsize=(12, 10))  # Initialize plot with size 12x10 inches

    # Plot each lapse year category
    for cat in unique_years:
        gdf_cat = gdf[gdf['life_category'] == cat]  # Filter claims by lapse year
        if gdf_cat.empty:
            continue  # Skip empty categories
        gdf_cat.plot(
            ax=ax,  # Use current axes
            color=cat_colors[cat],  # Set color for year
            edgecolor='black',  # Black borders
            linewidth=0.5,  # Border width
            alpha=0.7,  # Transparency
            label=cat  # Legend label (year)
        )

    # Plot property outlines
    gdf_outlines.plot(
        ax=ax,  # Use current axes
        facecolor='none',  # No fill
        edgecolor='black',  # Black borders
        linewidth=2,  # Thicker borders
        label='Property Outlines'  # Legend label
    )

    # Add north arrow
    ax.annotate('N', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='center', va='center')  # Add 'N' label
    ax.arrow(0.05, 0.90, 0, 0.05, transform=ax.transAxes, color='black', head_width=0.02, head_length=0.03)  # Add arrow

    # Add scale bar
    scalebar = AnchoredSizeBar(
        ax.transData,  # Data coordinates
        1000,  # 1 km scale
        '1 km',  # Label
        loc='lower right',  # Position
        pad=0.5,  # Padding
        color='black',  # Color
        frameon=False,  # Frame off
        size_vertical=100  # Vertical size
    )
    ax.add_artist(scalebar)  # Add scale bar to plot

    # Set plot title and labels
    plt.title('Projected Lapse Years for GSM Claims with Ongoing Redistributions - UTM Zone 17N')  # Set title
    plt.xlabel('Easting (meters)')  # X-axis label
    plt.ylabel('Northing (meters)')  # Y-axis label

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)  # Add dashed grid with transparency

    # Create legend elements
    legend_elements = [
        Patch(facecolor=cat_colors[year], edgecolor='black', alpha=0.7, label=year)
        for year in unique_years
    ]  # Create patches for each lapse year
    legend_elements.append(Patch(facecolor='none', edgecolor='black', linewidth=2, label='Property Outlines'))  # Add outlines patch
    ax.legend(handles=legend_elements, loc='upper right')  # Add legend to plot

    # Adjust layout
    plt.tight_layout()  # Optimize plot layout

    # Save plot with error handling
    try:
        plt.savefig(os.path.join(output_dir, 'gsm_claims_lapse_years_map.png'), dpi=300, bbox_inches='tight')  # Save plot as PNG
    except PermissionError as e:
        print(f"PermissionError: Unable to write to {os.path.join(output_dir, 'gsm_claims_lapse_years_map.png')}. {str(e)}")  # Log permission error
        print("Please ensure the directory is writable and no files are open in other applications.")  # Suggest fix
        raise  # Re-raise to halt execution
    plt.show()  # Display plot

# Purpose: Generate a summary bar plot of claims by project and lapse year with deficit annotations
def plot_summary_by_project(gdf: gpd.GeoDataFrame, unresolved: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Generate a bar plot showing the count of lapsed claims by lapse year and project, with years ascending to the right,
    and annotate each bar with the total deficit dollar value required to keep those claims active.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with life_category and project.
        unresolved (List[Dict[str, Any]]): Unresolved deficits list.
        output_dir (str): Output directory path.
    """
    # Define function to categorize claims by lapse year
    def get_life_category(final_expiry_date: pd.Timestamp) -> str:
        return str(final_expiry_date.year)  # Return year as string (e.g., '2032')

    # Apply lapse year category to each claim if not already present
    if 'life_category' not in gdf.columns:
        gdf['life_category'] = gdf['final_expiry_date'].apply(get_life_category)  # Assign lapse year categories

    # Create bar plot for count of lapsed claims
    plt.figure(figsize=(12, 6))
    order = sorted(gdf['life_category'].unique())  # Sort lapse years ascending
    ax = sns.countplot(data=gdf, x='life_category', hue='project', order=order)

    # Calculate total deficits for annotation
    if unresolved:
        unresolved_df = pd.DataFrame(unresolved)
        # Group by year and project to get total deficit
        deficit_grouped = unresolved_df.groupby(['year', 'project'])['deficit_needed'].sum().reset_index(name='total_deficit')
        # Convert year to string to match life_category
        deficit_grouped['year'] = deficit_grouped['year'].astype(str)

        # Get bar positions and heights
        bars = ax.patches
        bar_data = []
        for bar in bars:
            bar_data.append({
                'x': bar.get_x() + bar.get_width() / 2,  # Center of bar
                'height': bar.get_height(),  # Height of bar
                'year': order[int(bar.get_x() + bar.get_width() / 2)],  # Corresponding year
                'project': bar.get_label() if hasattr(bar, 'get_label') else None  # Project from hue
            })

        # Annotate bars with deficit values
        for bar in bar_data:
            year = bar['year']
            project = bar['project']
            if project and project in deficit_grouped['project'].values:
                deficit_row = deficit_grouped[(deficit_grouped['year'] == year) & (deficit_grouped['project'] == project)]
                if not deficit_row.empty:
                    deficit = deficit_row['total_deficit'].iloc[0]
                    # Format deficit for display (e.g., $1.2K, $1.2M)
                    if deficit >= 1_000_000:
                        deficit_text = f"${deficit/1_000_000:.1f}M"
                    elif deficit >= 1_000:
                        deficit_text = f"${deficit/1_000:.1f}K"
                    else:
                        deficit_text = f"${deficit:.0f}"
                    ax.text(
                        bar['x'], 
                        bar['height'] + 0.1,  # Slightly above bar
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
    try:
        plt.savefig(os.path.join(output_dir, 'claims_by_project.png'), dpi=300, bbox_inches='tight')
        print(f"Exported summary plot to '{os.path.join(output_dir, 'claims_by_project.png')}'")
    except PermissionError as e:
        print(f"PermissionError: Unable to write to {os.path.join(output_dir, 'claims_by_project.png')}. {str(e)}")
        print("Please ensure the directory is writable and no files are open in other applications.")
        raise
    plt.show()

# Purpose: Unit tests for key functions
def run_tests():
    """
    Run basic unit tests for reduce_expirations and format_expirations functions.
    """
    # Test reduce_expirations
    exp_list = [{'date': pd.Timestamp('2028-10-16'), 'amount': 1377.81}, {'date': pd.Timestamp('2038-10-16'), 'amount': 636.54}]
    original = copy.deepcopy(exp_list)
    pulled = reduce_expirations(exp_list, 1000)
    assert pulled == [{'date': pd.Timestamp('2028-10-16'), 'amount': 1000}], "reduce_expirations failed"
    assert len(exp_list) == 2, "reduce_expirations length failed"
    assert exp_list[0]['date'] == pd.Timestamp('2028-10-16') and abs(exp_list[0]['amount'] - 377.81) < 0.01, "reduce_expirations amount failed for 2028"
    assert exp_list[1]['date'] == pd.Timestamp('2038-10-16') and abs(exp_list[1]['amount'] - 636.54) < 0.01, "reduce_expirations amount failed for 2038"
    
    # Test format_expirations
    test_list = [{'date': pd.Timestamp('2028-10-16'), 'amount': 1377.81}, {'date': pd.Timestamp('2038-10-16'), 'amount': 0}]
    assert format_expirations(test_list) == '2028/10/16 (1377.81 $)', "format_expirations failed"
    assert format_expirations('2028/10/16 (1377.81 $)') == '2028/10/16 (1377.81 $)', "format_expirations string input failed"
    print("All unit tests passed!")

# Purpose: Main execution block to run the simulation and generate outputs
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Mining Claim Redistribution Simulation')
    parser.add_argument('--max-distance', type=float, default=CONFIG['MAX_DISTANCE'], help='Maximum distance for credit redistribution (meters)')
    parser.add_argument('--scoring-mode', type=str, choices=['earliest_expiry', 'distance_surplus'], default=CONFIG['SCORING_MODE'], help='Donor scoring mode')
    parser.add_argument('--csv-path', type=str, default=CONFIG['CSV_PATH'], help='Path to claims CSV file')
    parser.add_argument('--shp-path', type=str, default=CONFIG['SHP_PATH'], help='Path to claims shapefile')
    parser.add_argument('--outlines-shp', type=str, default=CONFIG['OUTLINES_SHP'], help='Path to outlines shapefile')
    parser.add_argument('--output-dir', type=str, default=CONFIG['OUTPUT_DIR'], help='Output directory path')
    parser.add_argument('--current-date', type=str, default=CONFIG['CURRENT_DATE'], help='Simulation start date (YYYY-MM-DD)')
    parser.add_argument('--config-path', type=str, default=None, help='Path to JSON config file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_path)
    config['MAX_DISTANCE'] = args.max_distance if args.max_distance != CONFIG['MAX_DISTANCE'] else config.get('MAX_DISTANCE', CONFIG['MAX_DISTANCE'])
    config['SCORING_MODE'] = args.scoring_mode if args.scoring_mode != CONFIG['SCORING_MODE'] else config.get('SCORING_MODE', CONFIG['SCORING_MODE'])
    config['CSV_PATH'] = args.csv_path if args.csv_path != CONFIG['CSV_PATH'] else config.get('CSV_PATH', CONFIG['CSV_PATH'])
    config['SHP_PATH'] = args.shp_path if args.shp_path != CONFIG['SHP_PATH'] else config.get('SHP_PATH', CONFIG['SHP_PATH'])
    config['OUTLINES_SHP'] = args.outlines_shp if args.outlines_shp != CONFIG['OUTLINES_SHP'] else config.get('OUTLINES_SHP', CONFIG['OUTLINES_SHP'])
    config['OUTPUT_DIR'] = args.output_dir if args.output_dir != CONFIG['OUTPUT_DIR'] else config.get('OUTPUT_DIR', CONFIG['OUTPUT_DIR'])
    config['CURRENT_DATE'] = args.current_date if args.current_date != CONFIG['CURRENT_DATE'] else config.get('CURRENT_DATE', CONFIG['CURRENT_DATE'])

    # Run unit tests
    run_tests()

    # Load and prepare data
    gdf, gdf_outlines = load_and_prepare_data(config['CSV_PATH'], config['SHP_PATH'], config['OUTLINES_SHP'])  # Load claims and outlines
    # Initialize simulation
    gdf = initialize_simulation(gdf)  # Initialize simulation columns
    # Precompute spatial data
    dist_matrix, sindex = precompute_spatial_data(gdf)  # Compute distance matrix and spatial index
    # Set current date
    current_date = date.fromisoformat(config['CURRENT_DATE'])  # Parse simulation start date
    # Run simulation
    log_table, unresolved = run_simulation(gdf, dist_matrix, sindex, current_date, config)  # Run simulation
    # Export results
    export_results(log_table, unresolved, gdf, config['OUTPUT_DIR'], current_date)  # Save results to CSV
    # Plot results (static)
    plot_results(gdf, gdf_outlines, config['OUTPUT_DIR'])  # Generate and save static map
    # Plot interactive map
    plot_interactive_map(gdf, log_table, unresolved, gdf_outlines, config['OUTPUT_DIR'])  # Generate and save interactive map
    # Plot summary by project
    plot_summary_by_project(gdf, unresolved, config['OUTPUT_DIR'])  # Generate and save summary bar plot

    # Print summary statistics
    print(f"Total claims analyzed: {len(gdf)}")  # Print total claims processed
    print(f"Average years of life: {gdf['years_of_life'].mean():.2f}")  # Print average life expectancy
    print(f"Minimum years of life: {gdf['years_of_life'].min():.2f}")  # Print minimum life expectancy
    print(f"Maximum years of life: {gdf['years_of_life'].max():.2f}")  # Print maximum life expectancy
    print(f"Exported full redistribution log to {os.path.join(config['OUTPUT_DIR'], 'full_redistribution_log.csv')}")
    print(f"Exported claims years of life to {os.path.join(config['OUTPUT_DIR'], 'claims_years_of_life.csv')}")
    print(f"Exported unresolved credits to {os.path.join(config['OUTPUT_DIR'], 'unresolved_credits_by_year_property.csv')}")
    print(f"Exported required spend pivot to {os.path.join(config['OUTPUT_DIR'], 'required_spend_pivot.csv')}")
    print(f"Exported claim expiration history to {os.path.join(config['OUTPUT_DIR'], 'claim_expiration_history.csv')}")
    print(f"Exported interactive map to {os.path.join(config['OUTPUT_DIR'], 'interactive_gsm_claims_map.html')}")
    print(f"Exported summary plot to {os.path.join(config['OUTPUT_DIR'], 'claims_by_project.png')}")
    print(f"Exported summary report to {os.path.join(config['OUTPUT_DIR'], 'simulation_report.txt')}")