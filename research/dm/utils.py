import numpy as np

CSV_FILE = './data/canfar.net_storage_vault_file_AstroDataCitationDOI_CISTI.CANFAR_25.0066_data_table_chimefrbcat2.csv'
GALACTIC_DM_MAP_FILE = './data/dm_mw_healpix_per_event.pkl'
GLADE_FILE = './data/glade.txt'

EXCLUDE_FIELD = 'excluded_flag'
REPEATER_NAME_FIELD = 'repeater_name'
DISPERSION_MEASURE_FIELD = 'dm_fitb'
DISPERSION_MEASURE_BACKUP_FIELD = 'bonsai_dm'
RIGHT_ASCENSION_FIELD = 'ra'
DECLENATION_FIELD = 'dec'
BONSAI_SNR_FIELD = 'bonsai_snr'
SUB_NUM_FIELD = 'sub_num'
GALACTIC_LATITUDE_FIELD = 'gb'
GALACTIC_LONGITUDE_FIELD = 'gl'

DEFAULT_RA_RES = 16
DEFAULT_DEC_RES = 16
SNR_THRESHOLD = 9
GALACTIC_LATITUDE_THRESHOLD = 10
NSIDE = 16
MIN_COUNT = 5

glade_columns = [
    "GLADE_id", "PGC_id", "GWGC_name", "HyperLEDA_name", "2MASS_name", 
    "WISExSCOS_name", "SDSS_name", "Object_type", "RA", "Dec", "Distance_Mpc",
    "Distance_err", "Distance_flag", "B_mag", "B_mag_err" 
]

def parse_area_args(args):
    """Helper function to extract RA, Dec, Area, and Radius from command args."""
    asc, dec, area, radius = None, None, 5.0, 0.5  # Default radius of 0.5 degrees
    if len(args) > 0:
        pairs = [arg.split('=') for arg in args]
        
        pos_pair = next((pair for pair in pairs if pair[0] == "pos"), None)
        if pos_pair:
            asc, dec = map(float, pos_pair[1].split(','))
            
        area_pair = next((pair for pair in pairs if pair[0] == "area"), None)
        if area_pair:
            area = float(area_pair[1])
            
        radius_pair = next((pair for pair in pairs if pair[0] == "radius"), None)
        if radius_pair:
            radius = float(radius_pair[1])
            
    return asc, dec, area, radius

def parse_range(args):
    distance_range = (0, np.inf)
    if len(args) > 0:
        pairs = [arg.split('=') for arg in args]

        range_pair = next((pair for pair in pairs if pair[0] == "range"), None)
        if range_pair:
            distance_range = range_pair[1].split(',')
            distance_range[0] = float(distance_range[0])
            distance_range[1] = float(distance_range[1]) if distance_range[1] != 'inf' else np.inf

    return distance_range