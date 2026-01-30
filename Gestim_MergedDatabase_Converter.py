import pandas as pd
import os
import csv
import stat
import re
from datetime import datetime, timedelta

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

# Define the translation dictionary for data values (excluding Title Type)
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

# Expected column headers (including new columns)
expected_columns = list(translations.values()) + ['Preferred Filing Date', 'Expiration Quarter-Year']

# Columns to process for comma-to-dot replacement and numeric conversion
numeric_columns = ['Polygon Area', 'Area (Ha)', 'Surpluses', 'Required Works', 'Required Rights']

# Define input and output paths
input_folder = r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files"
output_folder = r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Translated"
log_folder = r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Logs"
property_file = r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Property_to_Claim.csv"
input_files = [
    r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Translated\Gestim_Wallbridge_090126.xlsx",
    r"C:\Users\akoldewey\Documents\Python\Gestim_Database\Files\Translated\translated_Gestim_Midland_090126.csv"
]
output_file = os.path.join(output_folder, f"Gestim_Wallbridge_Midland_{datetime.now().strftime('%Y%m%d')}.csv")

# Create output and log folders if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)

# Possible encodings to try
possible_encodings = ['utf-8-sig', 'latin1', 'utf-8', 'utf-16', 'iso-8859-1']

# Function to detect delimiter
def detect_delimiter(file_path, encoding='latin1'):
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            sample = file.readline()
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            return delimiter
    except Exception as e:
        print(f"Could not detect delimiter for {file_path}: {e}")
        return ','  # Default to comma if detection fails

# Function to log problematic lines
def log_bad_lines(file_path, bad_lines):
    log_file = os.path.join(log_folder, f"bad_lines_{os.path.basename(file_path)}.log")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Problematic lines in {file_path}:\n")
        for line_num, line in bad_lines:
            f.write(f"Line {line_num}: {line}\n")
    print(f"Logged problematic lines to {log_file}")

# Function to log sample values for debugging
def log_sample_values(file_path, column, values, label):
    log_file = os.path.join(log_folder, f"sample_{label}_{os.path.basename(file_path)}.log")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Sample {label} values in {file_path} for column {column} (up to 10):\n")
        for value in values[:10]:
            f.write(f"{value}\n")
    print(f"Logged sample {label} values to {log_file}")

# Function to log unmatched Title Numbers
def log_unmatched_titles(file_path, unmatched_titles):
    log_file = os.path.join(log_folder, f"unmatched_titles_{os.path.basename(file_path)}.log")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Unmatched Title Numbers in {file_path} (showing up to 10):\n")
        for title in unmatched_titles[:10]:
            f.write(f"Title Number: {title}\n")
        if len(unmatched_titles) > 10:
            f.write(f"... and {len(unmatched_titles) - 10} more unmatched Title Numbers\n")
    print(f"Logged unmatched Title Numbers to {log_file}")

# Function to check and fix file permissions
def check_and_fix_permissions(file_path):
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

# Function to normalize strings for matching
def normalize_string(s):
    s = str(s).strip()  # Convert to string and strip whitespace
    s = re.sub(r'\s+', '', s)  # Remove all internal whitespace
    s = re.sub(r'^(TN|CDC)', '', s, flags=re.IGNORECASE)  # Remove prefixes
    s = s.lstrip('0') or '0'  # Remove leading zeros, ensure '0' if empty
    return s

# Function to calculate Preferred Filing Date (61 days before Expiration Date)
def calculate_preferred_filing_date(expiration_date):
    try:
        exp_date = pd.to_datetime(expiration_date, errors='coerce')
        if pd.isna(exp_date):
            return ''
        return (exp_date - timedelta(days=61)).strftime('%Y-%m-%d')
    except Exception:
        return ''

# Function to categorize Expiration Date into Quarter-Year
def categorize_quarter_year(expiration_date):
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

# Function to read and process a CSV or XLSX file
def read_and_process_file(file_path, expected_columns, is_excel=False):
    if not os.path.exists(file_path):
        print(f"Input file {file_path} does not exist")
        return None

    df = None
    if is_excel:
        try:
            df = pd.read_excel(
                file_path,
                dtype=str,
                keep_default_na=False
            )
            print(f"Successfully read {file_path} as Excel")
        except Exception as e:
            print(f"Failed to read {file_path} as Excel: {e}")
            return None
    else:
        delimiter = None
        for encoding in possible_encodings:
            try:
                delimiter = detect_delimiter(file_path, encoding)
                print(f"Detected delimiter '{delimiter}' for {file_path} with {encoding} encoding")
                break
            except UnicodeDecodeError:
                print(f"Failed to detect delimiter for {file_path} with {encoding} encoding")
                continue

        if delimiter is None:
            print(f"Could not detect delimiter for {file_path}. Using default comma.")
            delimiter = ','

        bad_lines = []
        for encoding in possible_encodings:
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    sep=delimiter,
                    quoting=csv.QUOTE_ALL,
                    header=0,
                    on_bad_lines='warn',
                    dtype=str,
                    keep_default_na=False
                )
                print(f"Successfully read {file_path} with {encoding} encoding and delimiter '{delimiter}'")
                break
            except UnicodeDecodeError:
                print(f"Failed to read {file_path} with {encoding} encoding")
                continue
            except pd.errors.ParserError as e:
                print(f"ParserError for {file_path} with {encoding} encoding and delimiter '{delimiter}': {e}")
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        lines = file.readlines()
                        bad_lines = [(i+1, line) for i, line in enumerate(lines) if line.count(delimiter) != len(df.columns)-1]
                        if bad_lines:
                            log_bad_lines(file_path, bad_lines)
                    df = pd.read_csv(
                        file_path,
                        encoding=encoding,
                        sep=delimiter,
                        quoting=csv.QUOTE_ALL,
                        dtype=str,
                        on_bad_lines='skip',
                        keep_default_na=False
                    )
                    print(f"Fallback: Read {file_path} as strings with {encoding} encoding and delimiter '{delimiter}'")
                    break
                except Exception as e:
                    print(f"Fallback failed for {file_path} with {encoding} encoding: {e}")
                    continue

        if df is None:
            print(f"Could not read {file_path} with any of the encodings: {possible_encodings}")
            return None

    # Remove completely empty rows
    df = df.dropna(how='all')
    if df.empty:
        print(f"No valid data in {file_path} after removing empty rows")
        return None

    # Rename columns using the translation dictionary
    df.columns = [translations.get(col, col) for col in df.columns]
    
    # Ensure all expected columns are present (excluding new columns)
    base_columns = [col for col in expected_columns if col not in ['Preferred Filing Date', 'Expiration Quarter-Year']]
    for col in base_columns:
        if col not in df.columns:
            df[col] = ''
    
    # Apply data value translations
    for column, mapping in data_translations.items():
        if column in df.columns:
            df[column] = df[column].replace(mapping)
            print(f"Applied data value translations to {column} in {file_path}")
    
    # Replace commas with dots and convert to numeric for specified columns
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add new columns
    if 'Expiration Date' in df.columns:
        df['Preferred Filing Date'] = df['Expiration Date'].apply(calculate_preferred_filing_date)
        df['Expiration Quarter-Year'] = df['Expiration Date'].apply(categorize_quarter_year)
        print(f"Added 'Preferred Filing Date' and 'Expiration Quarter-Year' columns to {file_path}")
    
    # Reorder columns to match expected_columns
    df = df[expected_columns]
    
    return df

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
        log_sample_values(property_file, 'CLAIM', property_df['CLAIM'].unique(), 'CLAIM')
        log_file = os.path.join(log_folder, f"property_to_claim_sample.csv")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"First 5 rows of Property_to_Claim.csv:\n")
            f.write(property_df.head().to_string())
        print(f"Logged first 5 rows of Property_to_Claim.csv to {log_file}")
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

# Process and merge input files
combined_df = None
for input_file in input_files:
    is_excel = input_file.endswith('.xlsx')
    df = read_and_process_file(input_file, expected_columns, is_excel=is_excel)
    if df is None:
        continue

    # Perform VLOOKUP-like operation to append Property column
    if claim_to_property and 'Title Number' in df.columns:
        try:
            df['Title Number'] = df['Title Number'].apply(normalize_string)
            log_sample_values(input_file, 'Title Number', df['Title Number'].unique(), 'Title_Number')
            log_file = os.path.join(log_folder, f"sample_{os.path.basename(input_file)}.csv")
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"First 5 rows of {os.path.basename(input_file)}:\n")
                f.write(df[['Title Number']].head().to_string())
            print(f"Logged first 5 rows of {os.path.basename(input_file)} to {log_file}")
            
            df['Property'] = df['Title Number'].map(claim_to_property).fillna('Unknown')
            print(f"Successfully applied VLOOKUP for Property column in {input_file}")
            
            unmatched = df[df['Property'] == 'Unknown']['Title Number'].unique()
            if len(unmatched) > 0:
                log_unmatched_titles(input_file, unmatched)
                print(f"Warning: {len(unmatched)} Title Numbers in {input_file} did not match any CLAIM in Property_to_Claim.csv")
            
            # Reorder columns to place Property after Map Sheet
            cols = df.columns.tolist()
            cols.remove('Property')
            cols.insert(1, 'Property')
            df = df[cols]
        except Exception as e:
            print(f"Failed to apply VLOOKUP for Property column in {input_file}: {e}")
            df['Property'] = 'Unknown'
            cols = df.columns.tolist()
            cols.remove('Property')
            cols.insert(1, 'Property')
            df = df[cols]
    else:
        print(f"Skipping VLOOKUP for {input_file}: Property_to_Claim.csv not loaded or Title Number column missing")
        df['Property'] = 'Unknown'
        cols = df.columns.tolist()
        cols.remove('Property')
        cols.insert(1, 'Property')
        df = df[cols]

    # Combine with previous DataFrames
    if combined_df is None:
        combined_df = df
    else:
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        print(f"Combined {os.path.basename(input_file)} with previous data. Total rows: {len(combined_df)}")

# Final deduplication and saving
if combined_df is not None:
    if 'Title Number' in combined_df.columns:
        combined_df['Title Number'] = combined_df['Title Number'].apply(normalize_string)
        combined_df = combined_df.drop_duplicates(subset='Title Number', keep='last')
        print(f"Deduplicated combined DataFrame to {len(combined_df)} unique rows")
    else:
        print("Warning: 'Title Number' column missing in combined DataFrame; no deduplication performed")

    # Save the combined CSV with UTF-8 BOM
    try:
        if check_and_fix_permissions(output_file):
            combined_df.to_csv(output_file, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
            print(f"Translated and combined file saved as: {output_file}")
        else:
            print(f"Cannot save {output_file}: Write permission not granted")
    except Exception as e:
        print(f"Failed to save {output_file}: {e}")
else:
    print("No valid input files processed. No output generated.")

print("Translation and combination complete!")