import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os

# File paths
claims_shapefile_path = r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Shapefile\gsm_claims_20250703.shp"
csv_path = r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Translated\Gestim_Wallbridge_Midland_20250909.csv"
outlines_shapefile_path = r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Shapefile\wmc_property_outlines.shp"
output_shapefile_path = r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Shapefile\gsm_claims_20250909.shp"

# Column renaming dictionary
column_map = {
    'Map Sheet': 'MapSheet',
    'Property': 'Property',
    'Canton/Seigneury Name': 'CantonName',
    'Canton/Seigneury Code': 'CantonCode',
    'Polygon Type': 'PolyType',
    'Range/Block/Plot': 'RangePlot',
    'Row/Block': 'RowBlock',
    'Column/Lot': 'ColLot',
    'Part': 'Part',
    'Polygon Area': 'PolyArea',
    'Title Type': 'TitleType',
    'Title Number': 'title_no',
    'Title Status': 'TitleStat',
    'Staking Date': 'StakeDate',
    'Registration Date': 'RegDate',
    'Expiration Date': 'ExpDate',
    'Number of Deadlines': 'NumDead',
    'Number of Renewals': 'NumRenew',
    'Area (Ha)': 'AreaHa',
    'Related Act(s)': 'RelActs',
    'Surpluses': 'Surplus',
    'Required Works': 'ReqWorks',
    'Required Rights': 'ReqRights',
    'Holder(s) (Name, Number, and Percentage)': 'Holders',
    'SMS Site Map Sheet': 'SMSMap',
    'SMS Site Number': 'SMSNum',
    'Renewal in Progress': 'RenewProg',
    'Works in Progress': 'WorkProg',
    'Title Transfer': 'TitleTran',
    'Description': 'Desc',
    'Location Comment': 'LocComment',
    'Constraint Comment': 'ConComment',
    'Conversion/Substitution of Exclusive Exploration Rights': 'ConvExpl',
    'Merger of Exclusive Exploration Rights': 'MergeExpl',
    'U3O8 Discovery': 'U3O8Disc',
    'Incompatible Territory': 'IncompTerr',
    'Administrative Region': 'AdminReg',
    'MRC': 'MRC',
    'Municipality': 'Municip',
    'Expiration Dates and Amounts': 'ExpDates'
}

try:
    # Read the claims shapefile
    claims_gdf = gpd.read_file(claims_shapefile_path)
    
    # Keep only 'title_no' and geometry
    claims_gdf = claims_gdf[['title_no', 'geometry']]
    
    # Convert 'title_no' to string
    claims_gdf['title_no'] = claims_gdf['title_no'].astype(str)
    
    # Read the CSV
    csv_df = pd.read_csv(csv_path)
    
    # Print original CSV column names
    print("Original CSV column names:", csv_df.columns.tolist())
    
    # Rename columns using the predefined mapping
    csv_df = csv_df.rename(columns=column_map)
    
    # Convert 'title_no' in CSV to string
    csv_df['title_no'] = csv_df['title_no'].astype(str)
    
    # Print renamed CSV column names
    print("Renamed CSV column names:", csv_df.columns.tolist())
    
    # Merge with inner join
    merged_gdf = claims_gdf.merge(csv_df, on='title_no', how='inner')
    
    # Print final column names
    print("Output shapefile column names:", merged_gdf.columns.tolist())
    
    # Check if output shapefile is accessible
    try:
        if os.path.exists(output_shapefile_path):
            os.remove(output_shapefile_path)  # Attempt to remove existing file
    except PermissionError:
        print(f"Error: Cannot overwrite {output_shapefile_path}. It is being used by another process. Please close any applications using this file.")
        raise
    
    # Save the merged shapefile
    merged_gdf.to_file(output_shapefile_path, driver='ESRI Shapefile')
    print(f"Shapefile saved to {output_shapefile_path}")
    print(f"Number of features: {len(merged_gdf)}")
    
    # Read the property outlines shapefile
    outlines_gdf = gpd.read_file(outlines_shapefile_path)
    
    # Ensure both GeoDataFrames use the same CRS
    if merged_gdf.crs != outlines_gdf.crs:
        outlines_gdf = outlines_gdf.to_crs(merged_gdf.crs)
    
    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot claims shapefile (bottom layer)
    merged_gdf.plot(ax=ax, color='lightblue', edgecolor='gray', alpha=0.7)
    
    # Plot property outlines on top
    outlines_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2)
    
    # Set plot title and labels
    ax.set_title('Claims and Property Outlines')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    # Show the plot
    plt.show()
    
    # Optionally save the plot (uncomment to use)
    # plt.savefig(r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\plot.png", dpi=300, bbox_inches='tight')

except FileNotFoundError as e:
    print(f"Error: A file was not found. Check paths:\nClaims: {claims_shapefile_path}\nCSV: {csv_path}\nOutlines: {outlines_shapefile_path}\nError: {str(e)}")
except PermissionError as e:
    print(f"Error: Cannot access output file. Ensure {output_shapefile_path} is not open in another program.\nError: {str(e)}")
except Exception as e:
    print(f"An error occurred: {str(e)}")