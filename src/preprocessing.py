"""
Applies second round of preprocessing to the maritime data. First round includes changes based on the AIS data that we received from Transport Canada, and is not available in this base code.
"""

import pandas as pd 
import yaml 

@staticmethod
def predict_glsl(lat: float, lon: float) -> int:
    """
    Predict if a port is within the Great Lakes-St. Lawrence Seaway (GLSL) region.

    Parameters
    ----------
    lat : float
        Latitude of the port
    lon : float
        Longitude of the port

    Returns
    -------
    int
        1 if port is within GLSL region, 0 otherwise
    """
    glsl_rectangle = {
        'east': -65.98,
        'north': 50.59,
        'south': 41.09,
        'west': -92.16
    }

    if glsl_rectangle['west'] <= lon <= glsl_rectangle['east'] and glsl_rectangle['south'] <= lat <= glsl_rectangle['north']:
        return 1
    return 0

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

input_path = config['paths']['unprocessed_ais_data_path']
output_path = config['paths']['processed_ais_data_path']

# Load the data
data = pd.read_parquet(input_path)

# Replace country name by region
region_maps = {
    'algeria': 'africa',
    'angola': 'africa',
    'antigua barbuda': 'north america',
    'argentina': 'south america',
    'aruba': 'north america',
    'australia': 'oceania',
    'bahamas': 'north america',
    'bahrain': 'asia',
    'bangladesh': 'asia',
    'barbados': 'north america',
    'belgium': 'europe',
    'belize': 'north america',
    'benin': 'africa',
    'bermuda': 'north america',
    'bonaire, st.eustatius and saba': 'north america',
    'brazil': 'south america',
    'british virgin is': 'north america',
    'brunei': 'asia',
    'bulgaria': 'europe',
    'cameroon': 'africa',
    'canada': 'north america',
    'cape verde': 'africa',
    'chile': 'south america',
    'china': 'asia',
    'colombia': 'south america',
    'comoros': 'africa',
    'congo': 'africa',
    'costa rica': 'north america',
    'croatia': 'europe',
    'cuba': 'north america',
    'curacao': 'north america',
    'cyprus': 'asia',
    'denmark': 'europe',
    'dominica': 'north america',
    'dominican republic': 'north america',
    'ecuador': 'south america',
    'egypt': 'africa',
    'estonia': 'europe',
    'faroe is': 'europe',
    'finland': 'europe',
    'france': 'europe',
    'gabon': 'africa',
    'gambia': 'africa',
    'germany': 'europe',
    'ghana': 'africa',
    'gibraltar': 'europe',
    'greece': 'europe',
    'greenland': 'north america',
    'grenada': 'north america',
    'guadeloupe': 'north america',
    'guatemala': 'north america',
    'guinea': 'africa',
    'guyana': 'south america',
    'haiti': 'north america',
    'honduras': 'north america',
    'hong kong': 'asia',
    'iceland': 'europe',
    'india': 'asia',
    'indonesia': 'asia',
    'iraq': 'asia',
    'ireland': 'europe',
    'isle of man': 'europe',
    'israel': 'asia',
    'italy': 'europe',
    'ivory coast': 'africa',
    'jamaica': 'north america',
    'japan': 'asia',
    'jordan': 'asia',
    'kenya': 'africa',
    'korea': 'asia',
    'kuwait': 'asia',
    'latvia': 'europe',
    'lebanon': 'asia',
    'liberia': 'africa',
    'libya': 'africa',
    'lithuania': 'europe',
    'madagascar': 'africa',
    'malaysia': 'asia',
    'malta': 'europe',
    'marshall islands': 'oceania',
    'martinique': 'north america',
    'mauritania': 'africa',
    'mauritius': 'africa',
    'mexico': 'north america',
    'monaco': 'europe',
    'montenegro': 'europe',
    'morocco': 'africa',
    'mozambique': 'africa',
    'myanmar': 'asia',
    'namibia': 'africa',
    'netherlands': 'europe',
    'new zealand': 'oceania',
    'nigeria': 'africa',
    'norway': 'europe',
    'oman': 'asia',
    'pakistan': 'asia',
    'panama': 'north america',
    'papua new guinea': 'oceania',
    'peru': 'south america',
    'philippines': 'asia',
    'poland': 'europe',
    'portugal': 'europe',
    'puerto rico': 'north america',
    'qatar': 'asia',
    'reunion': 'africa',
    'romania': 'europe',
    'russia': 'europe',
    'saudi arabia': 'asia',
    'senegal': 'africa',
    'sierra leone': 'africa',
    'singapore': 'asia',
    'sint maarten': 'north america',
    'slovenia': 'europe',
    'south africa': 'africa',
    'spain': 'europe',
    'sri lanka': 'asia',
    'st kitts nevis': 'north america',
    'st lucia': 'north america',
    'st pierre miquelon': 'north america',
    'st vincent grenadines': 'north america',
    'sudan': 'africa',
    'suriname': 'south america',
    'sweden': 'europe',
    'taiwan': 'asia',
    'thailand': 'asia',
    'togo': 'africa',
    'trinidad tobago': 'north america',
    'tunisia': 'africa',
    'turkey': 'asia',
    'ukraine': 'europe',
    'united arab emirates': 'asia',
    'united kingdom': 'europe',
    'uruguay': 'south america',
    'us virgin is': 'north america',
    'usa': 'north america',
    'venezuela': 'south america',
    'vietnam': 'asia',
    'yemen': 'asia'
}

# Create origin_region and destination_region columns
data['origin_region'] = data['od_origin_country'].str.lower().map(region_maps)
data['destination_region'] = data['od_destination_country'].str.lower().map(region_maps)

# Replace origin port: keep GLSL ports as is, replace non-GLSL with their region name
data['od_origin_port'] = data.apply(
    lambda row: row['od_origin_port']
    if predict_glsl(row['od_origin_port_latitude'], row['od_origin_port_longitude']) == 1
    else row['origin_region'].lower(),
    axis=1
)

# Replace destination port: keep GLSL ports as is, replace non-GLSL with their region name
data['od_destination_port'] = data.apply(
    lambda row: row['od_destination_port']
    if predict_glsl(row['od_destination_port_latitude'], row['od_destination_port_longitude']) == 1
    else row['destination_region'].lower(),
    axis=1
)

# Keep only the specified columns
columns_to_keep = ['od_origin_port', 'od_destination_port', 'track_avgsog', 'mt_dwt', 'od_hours_elapsed_origin_to_destination', 'od_origin_dwell_time', 'day_count']
data = data[columns_to_keep]

# Rename columns
col_name_mapping = {
    'od_origin_port': 'origin_port',
    'od_destination_port': 'destination_port',
    'track_avgsog': 'sog',
    'mt_dwt': 'deadweight',
    'od_hours_elapsed_origin_to_destination': 'duration',
    'od_origin_dwell_time': 'dwell_time',
}
data = data.rename(columns=col_name_mapping)

# Sort by descending day_count
data = data.sort_values('day_count', ascending=False)

# Save the processed data
data.to_parquet(output_path)

#save to csv
data.to_csv(output_path.replace('.parquet', '.csv'), index=False)



