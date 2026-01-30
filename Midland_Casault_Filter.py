import pandas as pd
import os
import csv
import re

# Define the translation dictionary for column headers
translations = {
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

# Define the translation dictionary for data values
data_translations = {
    'Title Status': {
        'Actif': 'Active'
    },
    'Related Act(s)': {
        'Oui': 'Yes',
        'Non': 'No'
    },
    'Renewal in Progress': {
        'Non': 'No'
    },
    'Works in Progress': {
        'Oui': 'Yes',
        'Non': 'No'
    },
    'Title Transfer': {
        'Oui': 'Yes',
        'Non': 'No'
    },
    'Conversion/Substitution of Exclusive Exploration Rights': {
        'Non': 'No'
    },
    'Merger of Exclusive Exploration Rights': {
        'Non': 'No'
    },
    'U3O8 Discovery': {
        'Non': 'No'
    },
    'Incompatible Territory': {
        'Non': 'No'
    },
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

# Columns to process for comma-to-dot replacement and numeric conversion
numeric_columns = ['Superficie Polygone', 'Superficie (Ha)', 'Excédents', 'Droits requis']

# Expected column headers
expected_columns = list(translations.keys())

# List of claim numbers to filter
claim_numbers = [
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

# Define input and output paths
input_file = r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Gestim_Midland_09012026.xlsx"
property_file = r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Property_to_Claim.csv"
output_folder = r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Translated"
output_file = os.path.join(output_folder, "translated_Gestim_Midland_090126.csv")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to normalize strings for matching
def normalize_string(s):
    s = str(s).strip()  # Convert to string and strip whitespace
    s = re.sub(r'\s+', '', s)  # Remove all internal whitespace
    s = re.sub(r'^(TN|CDC)', '', s, flags=re.IGNORECASE)  # Remove prefixes
    s = s.lstrip('0') or '0'  # Remove leading zeros, ensure '0' if empty
    return s

# Load the Property_to_Claim.csv file
try:
    property_df = pd.read_csv(
        property_file,
        encoding='latin1',
        quoting=csv.QUOTE_ALL,
        dtype=str,
        keep_default_na=False
    )
    if 'CLAIM' in property_df.columns and 'PROPERTY' in property_df.columns:
        property_df = property_df[['CLAIM', 'PROPERTY']].rename(columns={'PROPERTY': 'Property'})
        property_df['CLAIM'] = property_df['CLAIM'].apply(normalize_string)
        print(f"Loaded Property_to_Claim.csv with {len(property_df)} rows")
    else:
        print(f"Error: 'CLAIM' or 'PROPERTY' not found in {property_file}. Available columns: {list(property_df.columns)}")
        property_df = None
except Exception as e:
    print(f"Failed to read {property_file}: {e}")
    property_df = None

# Create a dictionary for VLOOKUP-like functionality
if property_df is not None:
    claim_to_property = dict(zip(property_df['CLAIM'], property_df['Property']))
    print(f"Created mapping dictionary with {len(claim_to_property)} unique CLAIM-to-Property mappings")
else:
    claim_to_property = {}
    print("No Property mapping available; proceeding without Property column")

# Process the input Excel file
if os.path.exists(input_file):
    try:
        # Read Excel file with expected column names
        df = pd.read_excel(
            input_file,
            names=expected_columns,
            header=0,
            dtype=str,
            keep_default_na=False
        )
        print(f"Successfully read {input_file}")
        
        # Verify 'No titre' column exists
        if 'No titre' not in df.columns:
            print(f"Error: 'No titre' column not found in {input_file}. Available columns: {list(df.columns)}")
        else:
            # Normalize 'No titre' values
            df['No titre'] = df['No titre'].apply(normalize_string)
            # Filter for specified claim numbers
            filtered_df = df[df['No titre'].isin(claim_numbers)]
            print(f"Filtered {len(filtered_df)} rows matching the specified claim numbers")
            
            # Append Property column
            if claim_to_property:
                filtered_df['Property'] = filtered_df['No titre'].map(claim_to_property).fillna('Unknown')
                print(f"Applied VLOOKUP for Property column")
                # Reorder columns to place Property as the second column
                cols = filtered_df.columns.tolist()
                cols.remove('Property')
                cols.insert(1, 'Property')
                filtered_df = filtered_df[cols]
            else:
                filtered_df['Property'] = 'Unknown'
                cols = filtered_df.columns.tolist()
                cols.remove('Property')
                cols.insert(1, 'Property')
                filtered_df = filtered_df[cols]
                print("No Property mapping available; set Property column to 'Unknown'")
            
            # Rename columns using the translation dictionary
            filtered_df.columns = [translations.get(col, col) for col in filtered_df.columns]
            print("Applied column header translations")
            
            # Apply data value translations
            for column, mapping in data_translations.items():
                if column in filtered_df.columns:
                    filtered_df[column] = filtered_df[column].replace(mapping)
                    print(f"Applied data value translations to {column}")
            
            # Replace commas with dots and convert to numeric for specified columns
            translated_numeric_columns = [translations[col] for col in numeric_columns if col in translations]
            for col in translated_numeric_columns:
                if col in filtered_df.columns:
                    filtered_df[col] = filtered_df[col].str.replace(',', '.', regex=False)
                    filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
                    print(f"Converted {col} to numeric")
            
            # Save the translated CSV with UTF-8 BOM
            try:
                filtered_df.to_csv(output_file, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
                print(f"Translated file saved as: {output_file}")
            except Exception as e:
                print(f"Failed to save {output_file}: {e}")
    except Exception as e:
        print(f"Failed to read {input_file}: {e}")
else:
    print(f"Input file {input_file} does not exist")

print("Translation and filtering complete!")