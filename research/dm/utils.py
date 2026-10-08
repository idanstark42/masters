CSV_FILE = './data/canfar.net_storage_vault_file_AstroDataCitationDOI_CISTI.CANFAR_25.0066_data_table_chimefrbcat2.csv'
GALACTIC_DM_MAP_FILE = './data/dm_mw_healpix_per_event.pkl'
GLADE_FILE = './data/glade.txt'
REGLADE_FILE = './data/regalade_v2.fits'

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