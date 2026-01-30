"""
Streamlit application for simulating mining claim renewals and credit redistributions.
Users can upload a CSV file, specify shapefile paths, adjust simulation parameters,
and view an interactive map with surplus and lapse year layers, a histogram, pivot table, and summary statistics.
"""
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import pandas as pd
import seaborn as sns
import os
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Optional, List, Dict, Any, Tuple
import re
import copy
import folium
from folium.plugins import Draw
from tqdm import tqdm
import streamlit as st
import tempfile
import shutil
import datetime
import io
st.set_page_config(page_title="Gestim Claim Redistribution Simulator", layout="wide")
# Default configuration
CONFIG = {
    'MAX_DISTANCE': 3900.0,
    'MAX_YEAR': 2100,
    'MAX_RENEWALS': 6,
    'SCORING_MODE': 'earliest_expiry',
    'SCORING_WEIGHTS': {'surplus': 0.3, 'distance': 0.7},
    'SHP_PATH': r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Shapefile\gsm_claims_20250703.shp",
    'OUTLINES_SHP': r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Shapefile\wmc_property_outlines.shp",
    'CURRENT_DATE': '2025-09-11'
}
# Purpose: Load and prepare data from CSV and shapefiles
def load_and_prepare_data(csv_path: str, shp_path: str, outlines_path: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Shapefile not found at: {shp_path}")
    if not os.path.exists(outlines_path):
        raise FileNotFoundError(f"Outlines shapefile not found at: {outlines_path}")
    gdf_claims = gpd.read_file(shp_path)
    df_csv = pd.read_csv(csv_path, encoding='utf-8')
    required_cols = ['Title Number', 'Expiration Date', 'Surpluses', 'Required Works', 'Property', 'Area (Ha)', 'Number of Deadlines', 'Number of Renewals', 'Expiration Dates and Amounts']
    missing_cols = [col for col in required_cols if col not in df_csv.columns]
    if missing_cols:
        raise ValueError(f"Missing required CSV columns: {missing_cols}")
    gdf_claims['title_no'] = gdf_claims['title_no'].astype(str).str.strip()
    df_csv['Title Number'] = df_csv['Title Number'].astype(str).str.strip()
    csv_titles = set(df_csv['Title Number'])
    shp_titles = set(gdf_claims['title_no'])
    if not csv_titles.intersection(shp_titles):
        raise ValueError("No matching title_no values between CSV and shapefile")
    if len(csv_titles.intersection(shp_titles)) < len(csv_titles):
        st.warning(f"{len(csv_titles) - len(csv_titles.intersection(shp_titles))} title_no values in CSV not found in shapefile")
    for col in ['project', 'Property', 'PROJECT', 'property']:
        if col in gdf_claims.columns:
            gdf_claims = gdf_claims.rename(columns={col: f'{col}_shp'})
    column_mapping = {
        'Title Number': 'title_no',
        'Expiration Date': 'expiry_date',
        'Surpluses': 'excess_work',
        'Required Works': 'required_work',
        'Property': 'project',
        'Area (Ha)': 'area_ha',
        'Number of Deadlines': 'terms_completed',
        'Number of Renewals': 'renewals_done',
        'Expiration Dates and Amounts': 'credit_expirations_raw'
    }
    df_csv = df_csv.rename(columns=column_mapping)
    for col in ['excess_work', 'required_work', 'area_ha']:
        df_csv[col] = df_csv[col].astype(str).str.replace(',', '.').replace(['None', ''], '0').astype(float).round(2)
    for col in ['terms_completed', 'renewals_done']:
        df_csv[col] = df_csv[col].fillna(0).astype(int)
    def parse_expirations(exp_str: str) -> List[Dict[str, Any]]:
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
                    st.warning(f"Could not parse expiration entry: {entry}")
            else:
                st.warning(f"Invalid expiration format: {entry}")
        expirations.sort(key=lambda x: x['date'])
        return expirations
    df_csv['credit_expirations'] = df_csv['credit_expirations_raw'].apply(parse_expirations)
    for idx, row in df_csv.iterrows():
        exp_sum = sum(exp['amount'] for exp in row['credit_expirations'])
        if abs(exp_sum - row['excess_work']) > 0.01:
            st.warning(f"For title_no {row['title_no']}, expiration sum {exp_sum} does not match excess_work {row['excess_work']}")
    gdf = gdf_claims.merge(
        df_csv[['title_no', 'expiry_date', 'excess_work', 'required_work', 'project', 'area_ha', 'terms_completed', 'renewals_done', 'credit_expirations']],
        on='title_no',
        how='inner',
        suffixes=('_shp', None)
    )
    if gdf.empty:
        raise ValueError("Merge resulted in an empty DataFrame. Check title_no matches between CSV and shapefile.")
    if 'project' not in gdf.columns:
        for col in ['project_shp', 'Property_shp', 'PROJECT_shp', 'property_shp']:
            if col in gdf.columns:
                gdf['project'] = gdf[col]
                st.warning(f"Using {col} as project column from shapefile.")
                break
        else:
            raise KeyError("Column 'project' not found in merged GeoDataFrame and no fallback project-like column found in shapefile.")
    if gdf.crs != "EPSG:2958":
        gdf = gdf.to_crs(epsg=2958)
    gdf = gdf[~gdf['project'].isin(['DETOUR EAST', 'Radisson JV', 'NANTEL', 'HWY 810 - PUISSEAUX', 'HWY 810 - RAYMOND', 'N2', 'FENELON BM 864'])].copy().reset_index(drop=True)
    gdf_outlines = gpd.read_file(outlines_path)
    if gdf_outlines.crs != "EPSG:2958":
        gdf_outlines = gdf_outlines.to_crs(epsg=2958)
    return gdf, gdf_outlines
# Purpose: Initialize the GeoDataFrame with simulation-specific columns
def initialize_simulation(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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
# Purpose: Calculate required work credits based on Quebec mining regulations
def get_required_work(term: int, area_ha: float, latitude: float) -> float:
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
# Purpose: Precompute spatial data for efficient distance calculations
@st.cache_data
def precompute_spatial_data(_gdf: gpd.GeoDataFrame) -> Tuple[np.ndarray, Any]:
    points = np.array([list(p.coords[0]) for p in _gdf.centroid])
    dist_matrix = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=-1)
    cent_gdf = gpd.GeoDataFrame(geometry=_gdf.centroid, crs=_gdf.crs)
    cent_gdf.index = _gdf.index
    sindex = cent_gdf.sindex
    return dist_matrix, sindex
# Purpose: Reduce credit amounts from the earliest expiration entries
def reduce_expirations(exp_list: List[Dict[str, Any]], amount: float) -> List[Dict[str, Any]]:
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
# Purpose: Format expiration list as a readable string
def format_expirations(exp_list: List[Dict[str, Any]]) -> str:
    if not isinstance(exp_list, list):
        return exp_list
    return '; '.join(f"{exp['date'].strftime('%Y/%m/%d')} ({exp['amount']:.2f} $)" for exp in exp_list if exp['amount'] > 0) or ''
# Purpose: Run the core simulation
def run_simulation(gdf: gpd.GeoDataFrame, dist_matrix: np.ndarray, sindex: Any, current_date: date, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    MAX_DISTANCE = config['MAX_DISTANCE']
    MAX_YEAR = config['MAX_YEAR']
    MAX_RENEWALS = config['MAX_RENEWALS']
    SCORING_MODE = config['SCORING_MODE']
    w1 = config['SCORING_WEIGHTS']['surplus']
    w2 = config['SCORING_WEIGHTS']['distance']
    log_table = []
    unresolved = []
    total_lost_credits = 0.0
    total_steps = len(gdf[gdf['active']]) * 2 # Rough estimate for progress bar
    progress_bar = st.progress(0)
    step = 0
    while True:
        active_claims = gdf[gdf['active']]
        if active_claims.empty:
            break
        next_date = active_claims['current_expiry_date'].min()
        if pd.isna(next_date) or next_date.year > MAX_YEAR:
            gdf.at[active_claims.index, 'active'] = False
            break
        for idx in active_claims.index:
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
            total_lost_credits += to_remove
            if abs(sum(exp['amount'] for exp in new_list) - gdf.at[idx, 'current_excess_work']) > 0.01:
                st.warning(f"For title_no {gdf.at[idx, 'title_no']}, expiration sum does not match current_excess_work")
            step += 1
            progress_bar.progress(min(step / total_steps, 1.0))
        expiring = active_claims[active_claims['current_expiry_date'] == next_date]
        expiring = expiring.assign(deficit=np.maximum(0, expiring['current_required_work'] - expiring['current_excess_work']))
        expiring = expiring.sort_values(by='deficit', ascending=True)
        for idx, row in expiring.iterrows():
            if row['renewals'] >= MAX_RENEWALS:
                unresolved.append({
                    'year': next_date.year,
                    'project': row['project'],
                    'deficit_needed': max(0, row['current_required_work'] - row['current_excess_work']),
                    'title_no': row['title_no']
                })
                total_lost_credits += row['current_excess_work']  # Add remaining credits as lost
                gdf.at[idx, 'active'] = False
                continue
            excess = gdf.at[idx, 'current_excess_work']
            required = gdf.at[idx, 'current_required_work']
            deficit = required - excess
            if deficit <= 0:
                original_exp = copy.deepcopy(gdf.at[idx, 'credit_expirations'])
                pulled_portions = reduce_expirations(gdf.at[idx, 'credit_expirations'], required)
                gdf.at[idx, 'current_excess_work'] -= required
                gdf.at[idx, 'current_expiry_date'] += relativedelta(years=2)
                gdf.at[idx, 'renewals'] += 1
                gdf.at[idx, 'current_term'] += 1
                gdf.at[idx, 'current_required_work'] = get_required_work(gdf.at[idx, 'current_term'], row['area_ha'], row['latitude'])
                gdf.at[idx, 'final_expiry_date'] = gdf.at[idx, 'current_expiry_date']
                log_entry = {
                    'action_type': 'renewal',
                    'title_no': row['title_no'],
                    'renewal_date': next_date,
                    'renewal_year': next_date.year,
                    'renewal_amount': required,
                    'pulled_expirations': format_expirations(pulled_portions),
                    'original_expirations': format_expirations(original_exp),
                    'updated_expirations': format_expirations(gdf.at[idx, 'credit_expirations'])
                }
                log_table.append(log_entry)
                if abs(sum(exp['amount'] for exp in gdf.at[idx, 'credit_expirations']) - gdf.at[idx, 'current_excess_work']) > 0.01:
                    st.warning(f"For title_no {row['title_no']}, expiration sum does not match current_excess_work after renewal")
                continue
            credits_needed = deficit
            buffer_geo = row['centroid'].buffer(MAX_DISTANCE)
            candidates_pos = sindex.query(buffer_geo, predicate='intersects')
            nearby = gdf.iloc[candidates_pos]
            nearby = nearby[
                (nearby['active']) &
                (nearby.index != idx) &
                (nearby['current_excess_work'] > nearby['current_required_work'])
            ]
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
                        if abs(sum(exp['amount'] for exp in gdf.at[idx, 'credit_expirations']) - gdf.at[idx, 'current_excess_work']) > 0.01:
                            st.warning(f"For title_no {row['title_no']}, expiration sum does not match current_excess_work after redistribution")
                        log_entry = {
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
                        }
                        log_table.append(log_entry)
                        credits_needed -= pull
                    if credits_needed <= 0:
                        break
            if credits_needed <= 0:
                original_exp = copy.deepcopy(gdf.at[idx, 'credit_expirations'])
                pulled_portions = reduce_expirations(gdf.at[idx, 'credit_expirations'], gdf.at[idx, 'current_required_work'])
                gdf.at[idx, 'current_excess_work'] -= gdf.at[idx, 'current_required_work']
                gdf.at[idx, 'current_expiry_date'] += relativedelta(years=2)
                gdf.at[idx, 'renewals'] += 1
                gdf.at[idx, 'current_term'] += 1
                gdf.at[idx, 'current_required_work'] = get_required_work(gdf.at[idx, 'current_term'], row['area_ha'], row['latitude'])
                gdf.at[idx, 'final_expiry_date'] = gdf.at[idx, 'current_expiry_date']
                log_entry = {
                    'action_type': 'renewal',
                    'title_no': row['title_no'],
                    'renewal_date': next_date,
                    'renewal_year': next_date.year,
                    'renewal_amount': gdf.at[idx, 'current_required_work'],
                    'pulled_expirations': format_expirations(pulled_portions),
                    'original_expirations': format_expirations(original_exp),
                    'updated_expirations': format_expirations(gdf.at[idx, 'credit_expirations'])
                }
                log_table.append(log_entry)
                if abs(sum(exp['amount'] for exp in gdf.at[idx, 'credit_expirations']) - gdf.at[idx, 'current_excess_work']) > 0.01:
                    st.warning(f"For title_no {row['title_no']}, expiration sum does not match current_excess_work after renewal")
            else:
                unresolved.append({
                    'year': next_date.year,
                    'project': row['project'],
                    'deficit_needed': credits_needed,
                    'title_no': row['title_no']
                })
                total_lost_credits += gdf.at[idx, 'current_excess_work']  # Add remaining credits as lost
                gdf.at[idx, 'active'] = False
            step += 1
            progress_bar.progress(min(step / total_steps, 1.0))
    progress_bar.progress(1.0)
    return log_table, unresolved, total_lost_credits
# Purpose: Export simulation results to CSV files
def export_results(log_table: List[Dict[str, Any]], unresolved: List[Dict[str, Any]], gdf: gpd.GeoDataFrame, output_dir: str, current_date: date, total_lost_credits: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(log_table).to_csv(os.path.join(output_dir, 'full_redistribution_log.csv'), index=False)
    pivot_df = pd.DataFrame()
    if unresolved:
        unresolved_df = pd.DataFrame(unresolved)
        grouped = unresolved_df.groupby(['year', 'project'])['deficit_needed'].sum().reset_index(name='total_money_needed')
        grouped['total_money_needed'] = grouped['total_money_needed'].clip(lower=0)
        grouped.to_csv(os.path.join(output_dir, 'unresolved_credits_by_year_property.csv'), index=False)
        current_year = current_date.year
        max_year = current_year + 20
        grouped_filtered = grouped[grouped['year'] <= max_year]
        if not grouped_filtered.empty:
            pivot_df = grouped_filtered.pivot(index='year', columns='project', values='total_money_needed').fillna(0)
            pivot_df['Total'] = pivot_df.sum(axis=1)
            pivot_df.to_csv(os.path.join(output_dir, 'required_spend_pivot.csv'))
    gdf['days_of_life'] = (gdf['final_expiry_date'] - pd.Timestamp(current_date)).dt.days
    gdf['years_of_life'] = gdf['days_of_life'] / 365.25
    gdf[['title_no', 'project', 'expiry_date', 'final_expiry_date', 'renewals', 'years_of_life']].to_csv(
        os.path.join(output_dir, 'claims_years_of_life.csv'), index=False
    )
    expiration_history = []
    for idx, row in gdf.iterrows():
        expiration_history.append({
            'title_no': row['title_no'],
            'project': row['project'],
            'final_expiry_date': row['final_expiry_date'],
            'credit_expirations': format_expirations(row['credit_expirations'])
        })
    pd.DataFrame(expiration_history).to_csv(os.path.join(output_dir, 'claim_expiration_history.csv'), index=False)
    report = [
        f"Simulation Summary Report",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Claims Analyzed: {len(gdf)}",
        f"Average Years of Life: {gdf['years_of_life'].mean():.2f}",
        f"Median Years of Life: {gdf['years_of_life'].median():.2f}",
        f"Minimum Years of Life: {gdf['years_of_life'].min():.2f}",
        f"Maximum Years of Life: {gdf['years_of_life'].max():.2f}",
        f"Total Credits Redistributed: {sum(entry['credits_pulled'] for entry in log_table if entry['action_type'] == 'redistribution'):.2f}",
        f"Total Credits Lost: {total_lost_credits:.2f}",
        f"Claims Lapsed: {len(unresolved)}"
    ]
    with open(os.path.join(output_dir, 'simulation_report.txt'), 'w') as f:
        f.write('\n'.join(report))
    summary_stats = {
        'total_claims': len(gdf),
        'avg_years_life': gdf['years_of_life'].mean(),
        'median_years_life': gdf['years_of_life'].median(),
        'min_years_life': gdf['years_of_life'].min(),
        'max_years_life': gdf['years_of_life'].max(),
        'credits_redistributed': sum(entry.get('credits_pulled', 0) for entry in log_table if entry['action_type'] == 'redistribution'),
        'num_renewals': sum(1 for entry in log_table if entry['action_type'] == 'renewal'),
        'num_redistributions': sum(1 for entry in log_table if entry['action_type'] == 'redistribution'),
        'claims_lapsed': len(unresolved),
        'total_unresolved_deficit': sum(entry.get('deficit_needed', 0) for entry in unresolved),
        'total_lost_credits': total_lost_credits
    }
    return pivot_df, summary_stats
# Purpose: Generate static map for download
def plot_results(gdf: gpd.GeoDataFrame, gdf_outlines: gpd.GeoDataFrame, output_dir: str) -> None:
    def get_life_category(final_expiry_date: pd.Timestamp) -> str:
        return str(final_expiry_date.year)
    if 'life_category' not in gdf.columns:
        gdf['life_category'] = gdf['final_expiry_date'].apply(get_life_category)
    unique_years = sorted(gdf['life_category'].unique())
    n_years = len(unique_years)
    if n_years > 0:
        colors = mcolors.LinearSegmentedColormap.from_list(
            'hot_to_cold', ['red', 'orange', 'yellow', 'green', 'blue'], N=n_years
        )(np.linspace(0, 1, n_years))
        cat_colors = {year: colors[i] for i, year in enumerate(unique_years)}
    else:
        cat_colors = {}
    fig, ax = plt.subplots(figsize=(12, 10))
    for cat in unique_years:
        gdf_cat = gdf[gdf['life_category'] == cat]
        if gdf_cat.empty:
            continue
        gdf_cat.plot(
            ax=ax,
            color=cat_colors[cat],
            edgecolor='black',
            linewidth=0.5,
            alpha=0.7,
            label=cat
        )
    gdf_outlines.plot(
        ax=ax,
        facecolor='none',
        edgecolor='black',
        linewidth=2,
        label='Property Outlines'
    )
    ax.annotate('N', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='center', va='center')
    ax.arrow(0.05, 0.90, 0, 0.05, transform=ax.transAxes, color='black', head_width=0.02, head_length=0.03)
    scalebar = AnchoredSizeBar(
        ax.transData,
        1000,
        '1 km',
        loc='lower right',
        pad=0.5,
        color='black',
        frameon=False,
        size_vertical=100
    )
    ax.add_artist(scalebar)
    plt.title('Projected Lapse Years for GSM Claims with Ongoing Redistributions - UTM Zone 17N')
    plt.xlabel('Easting (meters)')
    plt.ylabel('Northing (meters)')
    ax.grid(True, linestyle='--', alpha=0.7)
    legend_elements = [
        Patch(facecolor=cat_colors[year], edgecolor='black', alpha=0.7, label=year)
        for year in unique_years
    ]
    legend_elements.append(Patch(facecolor='none', edgecolor='black', linewidth=2, label='Property Outlines'))
    ax.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gsm_claims_lapse_years_map.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'gsm_claims_lapse_years_map.jpg'), dpi=300, bbox_inches='tight', format='jpg')
    plt.close(fig)
# Purpose: Generate interactive map with surplus and lapse year layers
def plot_interactive_map(gdf: gpd.GeoDataFrame, log_table: List[Dict[str, Any]], unresolved: List[Dict[str, Any]], gdf_outlines: gpd.GeoDataFrame, output_dir: str, show_surplus: bool = False) -> None:
    def get_life_category(final_expiry_date: pd.Timestamp) -> str:
        return str(final_expiry_date.year)
    if 'centroid' not in gdf.columns:
        gdf['centroid'] = gdf.geometry.centroid
    if 'life_category' not in gdf.columns and not show_surplus:
        gdf['life_category'] = gdf['final_expiry_date'].apply(get_life_category)
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
    if show_surplus:
        surplus_group = folium.FeatureGroup(name='Claims Surplus', show=True)
        def get_surplus_color(surplus: float) -> str:
            if surplus == 0:
                return 'grey'
            elif 0 < surplus <= 2500:
                return 'blue'
            elif 2500 < surplus <= 5000:
                return 'green'
            elif 5000 < surplus <= 10000:
                return 'yellow'
            elif 10000 < surplus <= 100000:
                return 'orange'
            else:
                return 'red'
        for idx, row in gdf_4326.iterrows():
            surplus = row['excess_work']
            color = get_surplus_color(surplus)
            tooltip = f"""
            Title No: {row['title_no']}<br>
            Project: {row['project']}<br>
            Surplus: ${surplus:.2f}
            """
            folium.GeoJson(
                row.geometry,
                style_function=lambda x, color=color: {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 0.5,
                    'fillOpacity': 0.7
                },
                tooltip=tooltip
            ).add_to(surplus_group)
        surplus_group.add_to(m)
        legend_html = '''
        <div style="position: fixed;
        bottom: 50px; left: 50px; width: 150px; height: auto;
        border:2px solid grey; z-index:9999; font-size:14px;
        background-color:white; padding: 10px;">
        <b>Surplus Legend ($)</b> <br>
        <i style="background:grey;width:20px;height:20px;float:left;margin-right:10px;"></i> 0.0-0.0<br>
        <i style="background:blue;width:20px;height:20px;float:left;margin-right:10px;"></i> 0.0-2500<br>
        <i style="background:green;width:20px;height:20px;float:left;margin-right:10px;"></i> 2500-5000<br>
        <i style="background:yellow;width:20px;height:20px;float:left;margin-right:10px;"></i> 5000-10000<br>
        <i style="background:orange;width:20px;height:20px;float:left;margin-right:10px;"></i> 10000-100000<br>
        <i style="background:red;width:20px;height:20px;float:left;margin-right:10px;"></i> >100000<br>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
    else:
        if 'life_category' not in gdf.columns:
            gdf['life_category'] = gdf['final_expiry_date'].apply(get_life_category)
        unique_years = sorted(gdf['life_category'].unique())
        n_years = len(unique_years)
        if n_years > 0:
            colors = mcolors.LinearSegmentedColormap.from_list(
                'hot_to_cold', ['red', 'orange', 'yellow', 'green', 'blue'], N=n_years
            )(np.linspace(0, 1, n_years))
            cat_colors = {year: mcolors.to_hex(colors[i]) for i, year in enumerate(unique_years)}
        else:
            cat_colors = {}
        claims_group = folium.FeatureGroup(name='Claims Lapse Years', show=True)
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
    outlines_group = folium.FeatureGroup(name='Property Outlines', show=True)
    folium.GeoJson(
        gdf_outlines_4326,
        style_function=lambda x: {'color': 'black', 'weight': 2, 'fillOpacity': 0}
    ).add_to(outlines_group)
    outlines_group.add_to(m)
    if not show_surplus:
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
                    st.warning(f"Skipping redistribution flow for donor {entry['donor_title_no']} or recipient {entry['insufficient_title_no']}")
        flow_group.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(os.path.join(output_dir, 'interactive_gsm_claims_map.html' if not show_surplus else 'surplus_map.html'))
# Purpose: Generate histogram
def plot_summary_by_project(gdf: gpd.GeoDataFrame, unresolved: List[Dict[str, Any]], output_dir: str) -> None:
    def get_life_category(final_expiry_date: pd.Timestamp) -> str:
        return str(final_expiry_date.year)
    if 'life_category' not in gdf.columns:
        gdf['life_category'] = gdf['final_expiry_date'].apply(get_life_category)
    lapsed_gdf = gdf[~gdf['active']].copy()
    if lapsed_gdf.empty:
        st.warning("No lapsed claims to plot in histogram.")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    order = sorted(lapsed_gdf['life_category'].unique())
    sns.countplot(data=lapsed_gdf, x='life_category', hue='project', order=order, ax=ax)
    plt.title('Number of Lapsed Claims by Project and Year')
    plt.xlabel('Lapse Year')
    plt.ylabel('Number of Lapsed Claims')
    plt.legend(title='Project')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'claims_by_project.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'claims_by_project.jpg'), dpi=300, bbox_inches='tight', format='jpg')
    plt.close(fig)
# Streamlit app
st.title("Gestim Claim Redistribution Simulator")
with st.form("simulation_form"):
    csv_file = st.file_uploader("Upload Claims CSV", type="csv")
    shp_path = st.text_input("Claims Shapefile Path", value=CONFIG['SHP_PATH'])
    outlines_path = st.text_input("Outlines Shapefile Path", value=CONFIG['OUTLINES_SHP'])
    max_distance = st.number_input("Max Distance (meters)", min_value=0.0, value=CONFIG['MAX_DISTANCE'])
    scoring_mode = st.selectbox("Scoring Mode", options=['earliest_expiry', 'distance_surplus'], index=0)
    surplus_weight = CONFIG['SCORING_WEIGHTS']['surplus']
    distance_weight = CONFIG['SCORING_WEIGHTS']['distance']
    if scoring_mode == 'distance_surplus':
        col1, col2 = st.columns(2)
        with col1:
            surplus_weight = st.number_input("Surplus Weight", min_value=0.0, max_value=1.0, value=CONFIG['SCORING_WEIGHTS']['surplus'], key="surplus_weight")
        with col2:
            distance_weight = st.number_input("Distance Weight", min_value=0.0, max_value=1.0, value=CONFIG['SCORING_WEIGHTS']['distance'], key="distance_weight")
    current_date_str = st.text_input("Current Date (YYYY-MM-DD)", value=CONFIG['CURRENT_DATE'])
    submit_button = st.form_submit_button("Run Simulation")
if csv_file and os.path.exists(shp_path) and os.path.exists(outlines_path):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
            tmp_csv.write(csv_file.getvalue())
            csv_path = tmp_csv.name
        output_dir = tempfile.mkdtemp()
        gdf, gdf_outlines = load_and_prepare_data(csv_path, shp_path, outlines_path)
        gdf = initialize_simulation(gdf)
        plot_interactive_map(gdf, [], [], gdf_outlines, output_dir, show_surplus=True)
        map_path = os.path.join(output_dir, 'surplus_map.html')
        with open(map_path, 'r') as f:
            html_data = f.read()
        st.subheader("Surplus Map (Pre-Simulation)")
        st.markdown('<div class="centered-content">', unsafe_allow_html=True)
        st.components.v1.html(html_data, height=600, scrolling=True)
        st.markdown('</div>', unsafe_allow_html=True)
        surplus_map_jpg = os.path.join(output_dir, 'gsm_claims_lapse_years_map.jpg')
        plot_results(gdf, gdf_outlines, output_dir)
        if os.path.exists(surplus_map_jpg):
            with open(surplus_map_jpg, 'rb') as f:
                st.download_button(
                    label="Download Surplus Map (JPG)",
                    data=f,
                    file_name="surplus_map.jpg",
                    mime="image/jpeg"
                )
        os.unlink(csv_path)
        shutil.rmtree(output_dir)
    except Exception as e:
        st.error(f"Error generating surplus map: {str(e)}")
if submit_button:
    if csv_file is None:
        st.error("Please upload a CSV file.")
    elif not os.path.exists(shp_path):
        st.error(f"Claims shapefile not found at: {shp_path}")
    elif not os.path.exists(outlines_path):
        st.error(f"Outlines shapefile not found at: {outlines_path}")
    elif scoring_mode == 'distance_surplus' and abs(surplus_weight + distance_weight - 1.0) > 0.01:
        st.error("Surplus Weight and Distance Weight must sum to 1.0 for distance_surplus mode.")
    elif max_distance < 100.0:
        st.warning("Max Distance is very low. Consider increasing it for meaningful redistributions.")
    else:
        try:
            current_date = datetime.datetime.strptime(current_date_str, '%Y-%m-%d').date()
        except ValueError:
            st.error("Invalid date format. Please use YYYY-MM-DD.")
        else:
            with st.spinner("Running simulation..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
                        tmp_csv.write(csv_file.getvalue())
                        csv_path = tmp_csv.name
                    output_dir = tempfile.mkdtemp()
                    config = CONFIG.copy()
                    config['CSV_PATH'] = csv_path
                    config['SHP_PATH'] = shp_path
                    config['OUTLINES_SHP'] = outlines_path
                    config['OUTPUT_DIR'] = output_dir
                    config['MAX_DISTANCE'] = max_distance
                    config['SCORING_MODE'] = scoring_mode
                    config['SCORING_WEIGHTS'] = {'surplus': surplus_weight, 'distance': distance_weight}
                    config['CURRENT_DATE'] = current_date_str
                    gdf, gdf_outlines = load_and_prepare_data(config['CSV_PATH'], config['SHP_PATH'], config['OUTLINES_SHP'])
                    gdf = initialize_simulation(gdf)
                    dist_matrix, sindex = precompute_spatial_data(gdf)
                    log_table, unresolved, total_lost_credits = run_simulation(gdf, dist_matrix, sindex, current_date, config)
                    pivot_df, summary_stats = export_results(log_table, unresolved, gdf, config['OUTPUT_DIR'], current_date, total_lost_credits)
                    plot_interactive_map(gdf, log_table, unresolved, gdf_outlines, config['OUTPUT_DIR'])
                    plot_summary_by_project(gdf, unresolved, config['OUTPUT_DIR'])
                    plot_results(gdf, gdf_outlines, config['OUTPUT_DIR'])
                    st.markdown(
                        """
                        <style>
                        .centered-content {
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            width: 100%;
                        }
                        .stDataFrame {
                            display: flex;
                            justify-content: center;
                            width: 100%;
                        }
                        table {
                            font-size: 18px !important;
                            text-align: center !important;
                        }
                        th, td {
                            text-align: center !important;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    st.subheader("Simulation Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Claims", summary_stats['total_claims'])
                    col2.metric("Avg Years Life", f"{summary_stats['avg_years_life']:.2f}")
                    col3.metric("Median Years Life", f"{summary_stats['median_years_life']:.2f}")
                    col4.metric("Min Years Life", f"{summary_stats['min_years_life']:.2f}")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Max Years Life", f"{summary_stats['max_years_life']:.2f}")
                    col2.metric("Credits Redistributed", f"{summary_stats['credits_redistributed']:.2f}")
                    col3.metric("Num Redistributions", summary_stats['num_redistributions'])
                    col4.metric("Num Renewals", summary_stats['num_renewals'])
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Claims Lapsed", summary_stats['claims_lapsed'])
                    col2.metric("Total Unresolved Deficit", f"{summary_stats['total_unresolved_deficit']:.2f}")
                    col3.metric("Total Credits Lost", f"{summary_stats['total_lost_credits']:.2f}")
                    map_path = os.path.join(config['OUTPUT_DIR'], 'interactive_gsm_claims_map.html')
                    with open(map_path, 'r') as f:
                        html_data = f.read()
                    st.subheader("Interactive Map")
                    st.markdown('<div class="centered-content">', unsafe_allow_html=True)
                    st.components.v1.html(html_data, height=600, scrolling=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    static_map_jpg = os.path.join(config['OUTPUT_DIR'], 'gsm_claims_lapse_years_map.jpg')
                    if os.path.exists(static_map_jpg):
                        with open(static_map_jpg, 'rb') as f:
                            st.download_button(
                                label="Download Static Map (JPG)",
                                data=f,
                                file_name="gsm_claims_lapse_years_map.jpg",
                                mime="image/jpeg"
                            )
                    hist_path = os.path.join(config['OUTPUT_DIR'], 'claims_by_project.jpg')
                    if os.path.exists(hist_path):
                        st.subheader("Claims by Project and Lapse Year")
                        st.markdown('<div class="centered-content">', unsafe_allow_html=True)
                        st.image(hist_path)
                        st.markdown('</div>', unsafe_allow_html=True)
                        with open(hist_path, 'rb') as f:
                            st.download_button(
                                label="Download Histogram (JPG)",
                                data=f,
                                file_name="claims_by_project.jpg",
                                mime="image/jpeg"
                            )
                    else:
                        st.warning("Histogram not generated, possibly due to no lapsed claims.")
                    if not pivot_df.empty:
                        st.subheader("Spend per Year and Property to Maintain Claims")
                        st.markdown('<div class="centered-content">', unsafe_allow_html=True)
                        st.dataframe(pivot_df.style.format("{:.2f}").set_properties(**{'text-align': 'center', 'font-size': '18px'}))
                        st.markdown('</div>', unsafe_allow_html=True)
                        csv_buffer = io.StringIO()
                        pivot_df.to_csv(csv_buffer, index=True)
                        st.download_button(
                            label="Download Pivot Table (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name="required_spend_pivot.csv",
                            mime="text/csv"
                        )
                    else:
                        st.write("No unresolved deficits to display in pivot table.")
                    os.unlink(csv_path)
                    shutil.rmtree(output_dir)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")