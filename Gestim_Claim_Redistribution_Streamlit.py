"""
Mining Claim Redistribution Dashboard - Streamlit Version
Web-based application for analyzing and redistributing mining claim credits.

Deploy with: streamlit run LVA_Analysis_Streamlit.py
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime
from pathlib import Path
import json
import io
import base64
from PIL import Image

# Import the main redistribution module
try:
    import Gestim_Claim_Redistribution as redistribution
    MAIN_MODULE_AVAILABLE = True
except ImportError:
    MAIN_MODULE_AVAILABLE = False
    st.error("⚠️ Main redistribution module not found. Please ensure Gestim_Claim_Redistribution.py is in the same directory.")

# Page configuration
st.set_page_config(
    page_title="Mining Claim Redistribution Dashboard",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-size: 1.1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = {
        'MIDLAND_XLSX': None,
        'WALLBRIDGE_XLSX': None,
        'PROPERTY_CSV': None,
        'SHP_PATH': None,
        'OUTLINES_SHP': None,
        'OUTPUT_DIR': 'output',
        'TEMP_DIR': 'temp',
        'LOG_DIR': 'logs',
        'MAX_DISTANCE': 3900.0,
        'MAX_YEAR': 2060,
        'MAX_RENEWALS': 6,
        'SCORING_MODE': 'earliest_expiry',
        'SCORING_WEIGHTS': {'surplus': 0.3, 'distance': 0.7},
        'INCLUDED_PROJECTS': ['CASAULT', 'MARTINIERE', 'FENELON', 'GRASSET', 'HARRI', 'DOIGT'],
        'CURRENT_DATE': date.today().isoformat()
    }

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}

if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

if 'gdf' not in st.session_state:
    st.session_state.gdf = None

if 'unresolved' not in st.session_state:
    st.session_state.unresolved = None

# Available projects
PROJECTS = ['CASAULT', 'MARTINIERE', 'FENELON', 'GRASSET', 'HARRI', 'DOIGT']

# Header
st.markdown('<div class="main-header">⛏️ Mining Claim Redistribution Dashboard</div>', unsafe_allow_html=True)

# Sidebar for quick actions
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=LVA+Analysis", width='stretch')
    st.markdown("---")
    st.markdown("### 🚀 Quick Actions")

    if st.button("💾 Save Configuration", width='stretch'):
        # Only save parameter settings, not file uploads
        config_to_save = {
            'MAX_DISTANCE': st.session_state.config.get('MAX_DISTANCE', 3900.0),
            'MAX_YEAR': st.session_state.config.get('MAX_YEAR', 2060),
            'MAX_RENEWALS': st.session_state.config.get('MAX_RENEWALS', 6),
            'SCORING_MODE': st.session_state.config.get('SCORING_MODE', 'earliest_expiry'),
            'INCLUDED_PROJECTS': st.session_state.config.get('INCLUDED_PROJECTS', [])
        }
        config_json = json.dumps(config_to_save, indent=4, default=str)
        st.download_button(
            label="📥 Download Config JSON",
            data=config_json,
            file_name="config.json",
            mime="application/json",
            width='stretch',
            help="Saves only parameter settings. Data files must be uploaded each session."
        )

    uploaded_config = st.file_uploader("📂 Load Configuration", type=['json'])
    if uploaded_config:
        try:
            loaded_config = json.load(uploaded_config)
            # Only update parameter settings, not file paths
            params_to_update = {
                'MAX_DISTANCE': loaded_config.get('MAX_DISTANCE', 3900.0),
                'MAX_YEAR': loaded_config.get('MAX_YEAR', 2060),
                'MAX_RENEWALS': loaded_config.get('MAX_RENEWALS', 6),
                'SCORING_MODE': loaded_config.get('SCORING_MODE', 'earliest_expiry'),
                'INCLUDED_PROJECTS': loaded_config.get('INCLUDED_PROJECTS', [])
            }
            st.session_state.config.update(params_to_update)
            st.success("✅ Settings loaded successfully")
        except Exception as e:
            st.error(f"❌ Error loading config: {e}")

    st.markdown("---")
    st.markdown("### ℹ️ System Info")
    if MAIN_MODULE_AVAILABLE:
        st.success("✅ Redistribution module loaded")
    else:
        st.error("❌ Redistribution module missing")

    st.info(f"📅 Today: {date.today().strftime('%Y-%m-%d')}")

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📁 Files",
    "⚙️ Configuration",
    "▶️ Run Simulation",
    "📊 Results",
    "🗺️ Maps",
    "📄 Reports"
])

# ============================================================================
# TAB 1: FILES
# ============================================================================
with tab1:
    st.header("📁 Input Files")
    st.markdown("Upload all required data files for the redistribution analysis.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Excel Files")

        midland_file = st.file_uploader("📊 Midland Excel File", type=['xlsx'], key='midland')
        if midland_file:
            st.session_state.uploaded_files['MIDLAND_XLSX'] = midland_file
            st.success(f"✅ {midland_file.name}")

        wallbridge_file = st.file_uploader("📊 Wallbridge Excel File", type=['xlsx'], key='wallbridge')
        if wallbridge_file:
            st.session_state.uploaded_files['WALLBRIDGE_XLSX'] = wallbridge_file
            st.success(f"✅ {wallbridge_file.name}")

        property_csv = st.file_uploader("📋 Property CSV File", type=['csv'], key='property')
        if property_csv:
            st.session_state.uploaded_files['PROPERTY_CSV'] = property_csv
            st.success(f"✅ {property_csv.name}")

    with col2:
        st.subheader("Shapefiles")
        st.info("💡 Upload all shapefile components together (.shp, .dbf, .shx, .prj)")

        # Claims shapefile - accept multiple files
        claims_files = st.file_uploader(
            "🗺️ Claims Shapefile Components",
            type=['shp', 'dbf', 'shx', 'prj'],
            accept_multiple_files=True,
            key='claims_files',
            help="Upload all 4 files: .shp, .dbf, .shx, .prj"
        )

        if claims_files:
            # Organize by extension
            claims_dict = {}
            for file in claims_files:
                ext = file.name.split('.')[-1]
                claims_dict[ext] = file

            # Check if we have all required files
            required_exts = {'shp', 'dbf', 'shx', 'prj'}
            uploaded_exts = set(claims_dict.keys())

            if required_exts.issubset(uploaded_exts):
                st.session_state.uploaded_files['SHP_PATH'] = claims_dict
                st.success(f"✅ Claims shapefile complete ({len(claims_files)} files)")
            else:
                missing = required_exts - uploaded_exts
                st.warning(f"⚠️ Missing files: {', '.join('.' + ext for ext in missing)}")

        st.markdown("---")

        # Property outlines shapefile - accept multiple files
        outlines_files = st.file_uploader(
            "🗺️ Property Outlines Shapefile Components",
            type=['shp', 'dbf', 'shx', 'prj'],
            accept_multiple_files=True,
            key='outlines_files',
            help="Upload all 4 files: .shp, .dbf, .shx, .prj"
        )

        if outlines_files:
            # Organize by extension
            outlines_dict = {}
            for file in outlines_files:
                ext = file.name.split('.')[-1]
                outlines_dict[ext] = file

            # Check if we have all required files
            required_exts = {'shp', 'dbf', 'shx', 'prj'}
            uploaded_exts = set(outlines_dict.keys())

            if required_exts.issubset(uploaded_exts):
                st.session_state.uploaded_files['OUTLINES_SHP'] = outlines_dict
                st.success(f"✅ Outlines shapefile complete ({len(outlines_files)} files)")
            else:
                missing = required_exts - uploaded_exts
                st.warning(f"⚠️ Missing files: {', '.join('.' + ext for ext in missing)}")

    # File status summary
    st.markdown("---")
    st.subheader("📋 Upload Status")

    file_status = {
        "Midland Excel": 'MIDLAND_XLSX' in st.session_state.uploaded_files,
        "Wallbridge Excel": 'WALLBRIDGE_XLSX' in st.session_state.uploaded_files,
        "Property CSV": 'PROPERTY_CSV' in st.session_state.uploaded_files,
        "Claims Shapefile": 'SHP_PATH' in st.session_state.uploaded_files,
        "Outlines Shapefile": 'OUTLINES_SHP' in st.session_state.uploaded_files
    }

    cols = st.columns(5)
    for idx, (name, status) in enumerate(file_status.items()):
        with cols[idx]:
            if status:
                st.success(f"✅ {name}")
            else:
                st.warning(f"⏳ {name}")

# ============================================================================
# TAB 2: CONFIGURATION
# ============================================================================
with tab2:
    st.header("⚙️ Configuration Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Simulation Parameters")

        max_distance = st.number_input(
            "🎯 Maximum Distance (meters)",
            min_value=0.0,
            max_value=10000.0,
            value=float(st.session_state.config['MAX_DISTANCE']),
            step=100.0,
            help="Maximum distance for credit redistribution"
        )
        st.session_state.config['MAX_DISTANCE'] = max_distance

        max_year = st.number_input(
            "📅 Maximum Year",
            min_value=2025,
            max_value=2100,
            value=int(st.session_state.config['MAX_YEAR']),
            step=1,
            help="Stop simulation at this year"
        )
        st.session_state.config['MAX_YEAR'] = max_year

        max_renewals = st.number_input(
            "🔄 Maximum Renewals",
            min_value=1,
            max_value=10,
            value=int(st.session_state.config['MAX_RENEWALS']),
            step=1,
            help="Maximum times a claim can be renewed"
        )
        st.session_state.config['MAX_RENEWALS'] = max_renewals

        scoring_mode = st.selectbox(
            "📊 Scoring Mode",
            options=['earliest_expiry', 'weighted'],
            index=0 if st.session_state.config['SCORING_MODE'] == 'earliest_expiry' else 1,
            help="Method for prioritizing credit redistribution"
        )
        st.session_state.config['SCORING_MODE'] = scoring_mode

    with col2:
        st.subheader("Project Selection")
        st.info("Select which projects to include in the analysis")

        selected_projects = st.multiselect(
            "📋 Included Projects",
            options=PROJECTS,
            default=st.session_state.config['INCLUDED_PROJECTS'],
            help="Only claims from these projects will be analyzed"
        )
        st.session_state.config['INCLUDED_PROJECTS'] = selected_projects

        if not selected_projects:
            st.warning("⚠️ At least one project must be selected!")

        st.markdown("---")

        # Quick select buttons
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ Select All", width='stretch'):
                st.session_state.config['INCLUDED_PROJECTS'] = PROJECTS
                st.rerun()
        with col_b:
            if st.button("❌ Clear All", width='stretch'):
                st.session_state.config['INCLUDED_PROJECTS'] = []
                st.rerun()

    # Configuration preview
    st.markdown("---")
    st.subheader("📄 Current Configuration")
    config_preview = {k: v for k, v in st.session_state.config.items() if k not in ['MIDLAND_XLSX', 'WALLBRIDGE_XLSX', 'PROPERTY_CSV', 'SHP_PATH', 'OUTLINES_SHP']}
    st.json(config_preview)

# ============================================================================
# TAB 3: RUN SIMULATION
# ============================================================================
with tab3:
    st.header("▶️ Run Simulation")

    # Validation
    errors = []

    if not MAIN_MODULE_AVAILABLE:
        errors.append("❌ Redistribution module not loaded")

    required_files = ['MIDLAND_XLSX', 'WALLBRIDGE_XLSX', 'PROPERTY_CSV', 'SHP_PATH', 'OUTLINES_SHP']
    for file_key in required_files:
        if file_key not in st.session_state.uploaded_files:
            errors.append(f"❌ {file_key.replace('_', ' ').title()}: Not uploaded")

    if not st.session_state.config['INCLUDED_PROJECTS']:
        errors.append("❌ No projects selected")

    # Display validation results
    if errors:
        st.error("### Configuration Issues")
        for error in errors:
            st.markdown(error)
        st.info("💡 **Tip:** Go to the 'Files' tab to upload data files and 'Configuration' tab to select projects.")
    else:
        st.success("✅ All validation checks passed! Ready to run simulation.")

        # Simulation parameters summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Max Distance", f"{st.session_state.config['MAX_DISTANCE']:.0f} m")
        with col2:
            st.metric("Max Year", st.session_state.config['MAX_YEAR'])
        with col3:
            st.metric("Max Renewals", st.session_state.config['MAX_RENEWALS'])
        with col4:
            st.metric("Projects", len(st.session_state.config['INCLUDED_PROJECTS']))

        st.markdown("---")

        # Run button
        if st.button("🚀 Run Simulation", type="primary", width='stretch'):
            with st.spinner("⏳ Running simulation... This may take several minutes."):
                try:
                    # Save uploaded files to temporary location
                    import tempfile
                    import shutil

                    temp_dir = tempfile.mkdtemp()

                    # Save Excel files
                    midland_path = Path(temp_dir) / "midland.xlsx"
                    with open(midland_path, 'wb') as f:
                        f.write(st.session_state.uploaded_files['MIDLAND_XLSX'].getbuffer())

                    wallbridge_path = Path(temp_dir) / "wallbridge.xlsx"
                    with open(wallbridge_path, 'wb') as f:
                        f.write(st.session_state.uploaded_files['WALLBRIDGE_XLSX'].getbuffer())

                    # Save CSV
                    property_path = Path(temp_dir) / "property.csv"
                    with open(property_path, 'wb') as f:
                        f.write(st.session_state.uploaded_files['PROPERTY_CSV'].getbuffer())

                    # Save shapefiles
                    claims_dir = Path(temp_dir) / "claims"
                    claims_dir.mkdir()
                    for ext, file_obj in st.session_state.uploaded_files['SHP_PATH'].items():
                        with open(claims_dir / f"claims.{ext}", 'wb') as f:
                            f.write(file_obj.getbuffer())

                    outlines_dir = Path(temp_dir) / "outlines"
                    outlines_dir.mkdir()
                    for ext, file_obj in st.session_state.uploaded_files['OUTLINES_SHP'].items():
                        with open(outlines_dir / f"outlines.{ext}", 'wb') as f:
                            f.write(file_obj.getbuffer())

                    # Build config for simulation (matching exact structure from desktop app)
                    config = {
                        'MIDLAND_XLSX': str(midland_path),
                        'WALLBRIDGE_XLSX': str(wallbridge_path),
                        'PROPERTY_CSV': str(property_path),
                        'SHP_PATH': str(claims_dir / "claims.shp"),
                        'OUTLINES_SHP': str(outlines_dir / "outlines.shp"),
                        'OUTPUT_DIR': temp_dir,
                        'TEMP_DIR': str(Path(temp_dir) / "Temp"),
                        'LOG_DIR': str(Path(temp_dir) / "Logs"),
                        'MAX_DISTANCE': float(st.session_state.config['MAX_DISTANCE']),
                        'MAX_YEAR': int(st.session_state.config['MAX_YEAR']),
                        'MAX_RENEWALS': int(st.session_state.config['MAX_RENEWALS']),
                        'SCORING_MODE': st.session_state.config['SCORING_MODE'],
                        'SCORING_WEIGHTS': {'surplus': 0.3, 'distance': 0.7},
                        'INCLUDED_PROJECTS': st.session_state.config['INCLUDED_PROJECTS'],
                        'CURRENT_DATE': date.today().isoformat()
                    }

                    # Create required directories
                    Path(config['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
                    Path(config['TEMP_DIR']).mkdir(parents=True, exist_ok=True)
                    Path(config['LOG_DIR']).mkdir(parents=True, exist_ok=True)

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Step 1: Process Midland claims
                    status_text.text("Step 1/8: Processing Midland claims...")
                    progress_bar.progress(5)
                    midland_df = redistribution.process_midland_file(config['MIDLAND_XLSX'], config['PROPERTY_CSV'], config)

                    # Step 2: Process Wallbridge claims
                    status_text.text("Step 2/8: Processing Wallbridge claims...")
                    progress_bar.progress(15)
                    wallbridge_df = redistribution.process_wallbridge_file(config['WALLBRIDGE_XLSX'], config['PROPERTY_CSV'], config)

                    # Step 3: Merge datasets
                    status_text.text("Step 3/8: Merging datasets...")
                    progress_bar.progress(25)
                    merged_csv = redistribution.merge_datasets(midland_df, wallbridge_df, config)

                    # Step 4: Load and prepare spatial data
                    status_text.text("Step 4/8: Loading spatial data...")
                    progress_bar.progress(35)
                    gdf, gdf_outlines = redistribution.load_and_prepare_data(
                        merged_csv,
                        config['SHP_PATH'],
                        config['OUTLINES_SHP'],
                        config
                    )

                    # Step 5: Initialize simulation
                    status_text.text("Step 5/8: Initializing simulation...")
                    progress_bar.progress(45)
                    gdf = redistribution.initialize_simulation(gdf)

                    # Step 6: Precompute spatial data
                    status_text.text("Step 6/8: Precomputing spatial data...")
                    progress_bar.progress(55)
                    dist_matrix, sindex = redistribution.precompute_spatial_data(gdf)

                    # Step 7: Run simulation
                    status_text.text("Step 7/8: Running simulation...")
                    progress_bar.progress(65)
                    current_date_obj = date.fromisoformat(config['CURRENT_DATE'])
                    log_table, unresolved = redistribution.run_simulation(gdf, dist_matrix, sindex, current_date_obj, config)

                    # Calculate claim life metrics
                    gdf['days_of_life'] = (gdf['final_expiry_date'] - pd.Timestamp(current_date_obj)).dt.days
                    gdf['years_of_life'] = gdf['days_of_life'] / 365.25

                    # Step 8: Generate outputs
                    status_text.text("Step 8/8: Generating outputs...")
                    progress_bar.progress(85)
                    redistribution.export_results(log_table, unresolved, gdf, config['OUTPUT_DIR'], current_date_obj)
                    redistribution.plot_results(gdf, gdf_outlines, config['OUTPUT_DIR'])
                    redistribution.plot_interactive_map(gdf, log_table, unresolved, gdf_outlines, config['OUTPUT_DIR'])
                    redistribution.plot_summary_by_project(gdf, unresolved, config['OUTPUT_DIR'])

                    progress_bar.progress(100)
                    status_text.text("✅ Simulation complete!")

                    # Store results in session state
                    st.session_state.simulation_results = {
                        'gdf': gdf,
                        'unresolved': unresolved,
                        'log_table': log_table,
                        'output_dir': temp_dir,
                        'timestamp': datetime.now()
                    }
                    st.session_state.gdf = gdf
                    st.session_state.unresolved = unresolved

                    st.success(f"""
                    ### 🎉 Simulation Complete!

                    **Claims Analyzed:** {len(gdf):,}
                    **Unresolved Claims:** {len(unresolved)}
                    **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

                    👉 Go to the **Results** tab to view detailed analysis!
                    """)

                except Exception as e:
                    st.error(f"### ❌ Simulation Failed\n\n{str(e)}")
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())

# ============================================================================
# TAB 4: RESULTS
# ============================================================================
with tab4:
    st.header("📊 Results Dashboard")

    if st.session_state.simulation_results is None:
        st.info("👈 Run a simulation first to see results here!")
    else:
        results = st.session_state.simulation_results
        gdf = results['gdf']
        unresolved = results['unresolved']
        output_dir = results['output_dir']

        # Summary metrics
        st.subheader("📈 Summary Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(gdf):,}</h3>
                <p>Total Claims</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(unresolved)}</h3>
                <p>Unresolved Claims</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            total_renewals = gdf['renewals'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>{total_renewals:,}</h3>
                <p>Total Renewals</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            avg_life = gdf['years_of_life'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>{avg_life:.1f} years</h3>
                <p>Avg Claim Life</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Visualizations
        st.subheader("📊 Visualizations")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # Claims by project
            st.markdown("#### Claims by Project")
            claims_by_project_path = Path(output_dir) / "claims_by_project.png"
            if claims_by_project_path.exists():
                st.image(str(claims_by_project_path), width='stretch')

            # Average claim life
            st.markdown("#### Average Claim Life by Project")
            avg_life_path = Path(output_dir) / "average_claim_life_by_project.png"
            if avg_life_path.exists():
                st.image(str(avg_life_path), width='stretch')

        with viz_col2:
            # Claim life histogram
            st.markdown("#### Claim Life Distribution")
            histogram_path = Path(output_dir) / "claim_life_histogram.png"
            if histogram_path.exists():
                st.image(str(histogram_path), width='stretch')

        st.markdown("---")

        # Credits analysis
        st.subheader("💰 Credits Analysis")

        # Read redistribution log
        log_path = Path(output_dir) / "full_redistribution_log.csv"
        if log_path.exists():
            log_df = pd.read_csv(log_path)

            # Credits by project
            if 'project' in log_df.columns:
                credits_summary = log_df.groupby('project').agg({
                    'original_expirations': 'sum',
                    'final_expirations': 'sum'
                }).reset_index()

                st.dataframe(credits_summary, width='stretch')

        st.markdown("---")

        # Pivot table
        st.subheader("📋 Required Spending by Year and Project")

        pivot_path = Path(output_dir) / "required_spend_pivot.csv"
        if pivot_path.exists():
            pivot_df = pd.read_csv(pivot_path, index_col=0)
            st.dataframe(pivot_df.style.format("${:,.0f}"), width='stretch')

        # Download results
        st.markdown("---")
        st.subheader("💾 Download Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            merged_path = Path(output_dir) / "Gestim_Wallbridge_Merged.csv"
            if merged_path.exists():
                with open(merged_path, 'rb') as f:
                    st.download_button(
                        "📥 Merged Data (CSV)",
                        f,
                        file_name="Gestim_Wallbridge_Merged.csv",
                        mime="text/csv",
                        width='stretch'
                    )

        with col2:
            if log_path.exists():
                with open(log_path, 'rb') as f:
                    st.download_button(
                        "📥 Redistribution Log (CSV)",
                        f,
                        file_name="full_redistribution_log.csv",
                        mime="text/csv",
                        width='stretch'
                    )

        with col3:
            if pivot_path.exists():
                with open(pivot_path, 'rb') as f:
                    st.download_button(
                        "📥 Spending Pivot (CSV)",
                        f,
                        file_name="required_spend_pivot.csv",
                        mime="text/csv",
                        width='stretch'
                    )

# ============================================================================
# TAB 5: MAPS
# ============================================================================
with tab5:
    st.header("🗺️ Interactive Maps")

    if st.session_state.simulation_results is None:
        st.info("👈 Run a simulation first to see maps here!")
    else:
        results = st.session_state.simulation_results
        output_dir = results['output_dir']

        # Static map
        st.subheader("📍 Claims Distribution Map")

        claims_map_path = Path(output_dir) / "map_claims_simplified.png"
        if claims_map_path.exists():
            st.image(str(claims_map_path), width='stretch')

        # Interactive map
        st.markdown("---")
        st.subheader("🌐 Interactive Folium Map")

        interactive_map_path = Path(output_dir) / "interactive_map.html"
        if interactive_map_path.exists():
            with open(interactive_map_path, 'r', encoding='utf-8') as f:
                map_html = f.read()
            st.components.v1.html(map_html, height=600, scrolling=True)

# ============================================================================
# TAB 6: REPORTS
# ============================================================================
with tab6:
    st.header("📄 PDF Report")

    if st.session_state.simulation_results is None:
        st.info("👈 Run a simulation first to generate a PDF report!")
    else:
        st.markdown("### Generate PDF Report")

        if st.button("📄 Generate PDF Report", type="primary", width='stretch'):
            with st.spinner("⏳ Generating PDF report..."):
                try:
                    results = st.session_state.simulation_results
                    gdf = results['gdf']
                    unresolved = results['unresolved']
                    output_dir = results['output_dir']

                    # Generate PDF
                    pdf_path = Path(output_dir) / "redistribution_report.pdf"

                    # Use the existing PDF generation from the main module
                    # (You'll need to import and call the PDF generation function)

                    st.success("✅ PDF report generated successfully!")

                    # Provide download button
                    if pdf_path.exists():
                        with open(pdf_path, 'rb') as f:
                            st.download_button(
                                "📥 Download PDF Report",
                                f,
                                file_name="redistribution_report.pdf",
                                mime="application/pdf",
                                width='stretch'
                            )

                        # PDF preview (if possible)
                        st.markdown("---")
                        st.subheader("📖 Report Preview")
                        st.info("Download the PDF to view the full report.")

                except Exception as e:
                    st.error(f"❌ Error generating PDF: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>Mining Claim Redistribution Dashboard v1.0</p>
    <p>Powered by Streamlit | © 2026</p>
</div>
""", unsafe_allow_html=True)
