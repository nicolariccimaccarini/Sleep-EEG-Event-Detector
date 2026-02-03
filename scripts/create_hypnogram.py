import pandas as pd
import re

def parse_time_to_seconds(time_str):
    """
    Converts a time string to seconds.
    Supported formats:
    - 1h5'32'' (hours, minutes, seconds)
    - 1h5' (hours, minutes)
    - 5'32'' (minutes, seconds)
    - 5' (minutes only)
    - 532 (minutes and seconds without quotes, e.g. 5'32)
    """
    if pd.isna(time_str) or time_str == '':
        return None
    
    time_str = str(time_str).strip()
    
    # Pattern for hours + minutes + seconds: 1h5'32''
    match = re.match(r"(\d+)h(\d+)'(\d+)''", time_str)
    if match:
        hours, minutes, secs = map(int, match.groups())
        return hours * 3600 + minutes * 60 + secs
    
    # Pattern for hours + minutes: 1h5'
    match = re.match(r"(\d+)h(\d+)'", time_str)
    if match:
        hours, minutes = map(int, match.groups())
        return hours * 3600 + minutes * 60
    
    # Pattern for minutes + seconds: 5'32''
    match = re.match(r"(\d+)'(\d+)''", time_str)
    if match:
        minutes, secs = map(int, match.groups())
        return minutes * 60 + secs
    
    # Pattern for minutes only: 5'
    match = re.match(r"(\d+)'$", time_str)
    if match:
        minutes = int(match.group(1))
        return minutes * 60
    
    # Pattern for minutes and seconds without quotes: 427 -> 4'27
    match = re.match(r"(\d)(\d{2})$", time_str)
    if match:
        minutes, secs = map(int, match.groups())
        return minutes * 60 + secs
    
    return None

def format_date(date_str):
    """
    Converts date from dd/mm/yyyy format to ddmmmyyyy format.
    Example: 7/1/2020 -> 07gen2020
    """
    # Italian month mapping
    month_map = {
        1: 'gen', 2: 'feb', 3: 'mar', 4: 'apr',
        5: 'mag', 6: 'giu', 7: 'lug', 8: 'ago',
        9: 'set', 10: 'ott', 11: 'nov', 12: 'dic'
    }
    
    # Handle malformed dates like 10/042024
    date_str = date_str.replace('/0', '/').strip()
    
    # Parse date
    try:
        # Try dd/mm/yyyy format
        parts = date_str.split('/')
        if len(parts) == 3:
            day = int(parts[0])
            month = int(parts[1])
            year = int(parts[2])
            
            return f"{day:02d}{month_map[month]}{year}"
    except:
        pass
    
    return date_str

def format_name(name, date_str):
    """
    Formats the name in the required format: NAME^SURNAMEddmmmyyyy
    """
    # Split name and surname
    parts = name.strip().split()
    if len(parts) >= 2:
        surname = parts[0]
        first_name = ' '.join(parts[1:])
        formatted_date = format_date(date_str)
        return f"{first_name}^{surname}{formatted_date}"
    return name

def convert_database_to_hypnogram(input_file, output_file):
    """
    Converts the database CSV to hypnogram format.
    """
    # Read CSV
    df = pd.read_csv(input_file)
    
    # List to store hypnogram rows
    hypnogram_rows = []
    
    # Process each row
    for idx, row in df.iterrows():
        name = row['NOME']
        date = row['DATA REGISTRAZIONE']
        
        # Format name
        formatted_name = format_name(name, date)
        
        # Process first NREM phase (columns 3 and 4)
        start1 = parse_time_to_seconds(row.iloc[3])
        end1 = parse_time_to_seconds(row.iloc[4])
        
        if start1 is not None and end1 is not None:
            hypnogram_rows.append({
                'nome': formatted_name,
                'start_time(s)': start1,
                'end_time(s)': end1
            })
        
        # Process second NREM phase (columns 5 and 6), if present
        if len(row) > 5:
            start2 = parse_time_to_seconds(row.iloc[5])
            end2 = parse_time_to_seconds(row.iloc[6])
            
            if start2 is not None and end2 is not None:
                hypnogram_rows.append({
                    'nome': formatted_name,
                    'start_time(s)': start2,
                    'end_time(s)': end2
                })
    
    # Create hypnogram DataFrame
    hypnogram_df = pd.DataFrame(hypnogram_rows)
    
    # Save CSV
    hypnogram_df.to_csv(output_file, index=False)
    print(f"Hypnogram saved to: {output_file}")
    print(f"Total rows processed: {len(hypnogram_rows)}")
    
    # Display first rows as example
    print("\nFirst rows of generated file:")
    print(hypnogram_df.head(10).to_string())

if __name__ == "__main__":
    input_file = "DATABASE_POP.csv"
    output_file = "hypnogram.csv"
    
    convert_database_to_hypnogram(input_file, output_file)