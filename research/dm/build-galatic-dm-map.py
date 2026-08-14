import numpy as np
import csv
import pickle
from astropy_healpix import HEALPix
from mwprop.nemod.NE2025 import ne2025
from astropy.coordinates import SkyCoord
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

EVENTS_CSV_FILE = './canfar.net_storage_vault_file_AstroDataCitationDOI_CISTI.CANFAR_25.0066_data_table_chimefrbcat2.csv'
RIGHT_ASCENSION_FIELD = 'ra'
DECLENATION_FIELD = 'dec'

def compute_pix(ra, dec):
  coord = SkyCoord(ra, dec, unit='deg', frame='galactic')
  l = coord.l.degree
  b = coord.b.degree
  try:
    Dk, Dv, Du, Dd = ne2025(ldeg=l, bdeg=b, dmd=1e6, ndir=1, classic=False)
    return float(Dv['DM'])
  except Exception as e:
    print(f"Error occurred while computing DM for ({ra}, {dec}): {e}")
    return np.nan

def _init_pool():
  # avoids repeated imports overhead in workers (optional but cleaner)
  pass

if __name__ == "__main__":
  print("Building galactic DM map...")
  print("Opening events CSV file: {}".format(EVENTS_CSV_FILE))
  with open(EVENTS_CSV_FILE, mode='r', encoding='utf-8') as file:
    dict_reader = csv.DictReader(file)
    events = list(dict_reader)
    print("Loaded {} events from {}".format(len(events), EVENTS_CSV_FILE))
  
    ras = [float(event[RIGHT_ASCENSION_FIELD]) for event in events]
    decs = [float(event[DECLENATION_FIELD]) for event in events]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
      results = list(
        tqdm(
          executor.map(compute_pix, ras, decs),
          total=len(events),
          ncols=80,
          unit="pix",
          desc="Building HEALPix DM map"
        )
      )

  dm_map = { f"{event[RIGHT_ASCENSION_FIELD]}_{event[DECLENATION_FIELD]}": dm_value for event, dm_value in zip(events, results) }

  with open("dm_mw_healpix_per_event.pkl", "wb") as f:
    pickle.dump(dm_map, f)