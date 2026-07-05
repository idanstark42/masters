import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy_healpix import HEALPix
from mwprop.nemod.NE2025 import ne2025
import tqdm
import pickle
from astropy_healpix import HEALPix


CSV_FILE = './canfar.net_storage_vault_file_AstroDataCitationDOI_CISTI.CANFAR_25.0066_data_table_chimefrbcat2.csv'

DISPERSION_MEASURE_FIELD = 'dm_fitb'
DISPERSION_MEASURE_BACKUP_FIELD = 'bonsai_dm'
RIGHT_ASCENSION_FIELD = 'ra'
DECLENATION_FIELD = 'dec'
BONSAI_SNR_FIELD = 'bonsai_snr'

DEFAULT_RA_RES = 4
DEFAULT_DEC_RES = 4
SNR_THRESHOLD = 9
GALACTIC_LATITUDE_THRESHOLD = 10

# -----------------------------
# CORE FUNCTIONS
# -----------------------------
def keys(dict_reader, args):
  print(dict_reader.fieldnames)

def event(dict_reader, args):
  index = args[0]
  rows = list(dict_reader)
  print(rows[int(index)])

def add_dm_exc(row, galactic_dm_map):
  ac = float(row[RIGHT_ASCENSION_FIELD])
  dec = float(row[DECLENATION_FIELD])

  dm_obs = float(row[DISPERSION_MEASURE_FIELD] or row[DISPERSION_MEASURE_BACKUP_FIELD])

  coord = SkyCoord(ra=ac*u.deg, dec=dec*u.deg, frame='icrs')
  coord_gal = coord.galactic

  # interpolate MW DM from HEALPix map
  pix = hp.skycoord_to_healpix(coord_gal)
  dm_mw = galactic_dm_map[pix]

  row["dm_exc"] = dm_obs - dm_mw
  return row

# runs all quality cuts and returns True if the event passes all cuts
def quality_event (row, galactic_dm_map):
  ac = row[RIGHT_ASCENSION_FIELD]
  dec = row[DECLENATION_FIELD]

  if float(row[BONSAI_SNR_FIELD]) < SNR_THRESHOLD:
    return False

  b = SkyCoord(
    ra=float(ac)*u.degree,
    dec=float(dec)*u.degree,
    frame='icrs'
  ).galactic.b.degree

  if abs(b) < GALACTIC_LATITUDE_THRESHOLD:
    return False

  row = add_dm_exc(row, galactic_dm_map)

  if row["dm_exc"] <= 0:
    return False

  return True

def heatmap(dict_reader, args):
  print("Loading Galactic DM map...")
  with open("dm_mw_healpix_ns16.pkl", "rb") as f:
    galactic_dm_map = pickle.load(f)
    print("Generating heatmap...")
    right_ascention_res = DEFAULT_RA_RES
    declenation_res = DEFAULT_DEC_RES

    if len(args) > 1:
      right_ascention_res, declenation_res = map(int, args[1].split(':'))

    values = []
    ras = []
    decs = []

    for row in tqdm.tqdm(dict_reader, desc="Processing events", ncols=1000, unit="event"):
      if not quality_event(row, galactic_dm_map):
        continue
      try:
        ras.append(float(row[RIGHT_ASCENSION_FIELD]))
        decs.append(float(row[DECLENATION_FIELD]))
        values.append(float(row[DISPERSION_MEASURE_FIELD] or row[DISPERSION_MEASURE_BACKUP_FIELD]))
      except (ValueError, KeyError):
        pass

    ra_edges = np.linspace(0, 360, right_ascention_res + 1)
    dec_edges = np.linspace(0, 90, declenation_res + 1)

    sums, _, _ = np.histogram2d(
      decs,
      ras,
      bins=[dec_edges, ra_edges],
      weights=values
    )

    counts, _, _ = np.histogram2d(
      decs,
      ras,
      bins=[dec_edges, ra_edges]
    )

    averages = np.divide(
      sums,
      counts,
      out=np.full_like(sums, np.nan),
      where=counts > 0
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(
      averages,
      origin='lower',
      aspect='auto',
      extent=[0, 360, 0, 90]
    )
    plt.colorbar(label='Dispersion Measure')
    plt.xlabel('Right Ascension (deg)')
    plt.ylabel('Declenation (deg)')
    plt.title('Average Dispersion Measure Heatmap')
    plt.show()

COMMANDS = {
  "keys": lambda x, a: keys(x, a),
  "event": lambda x, a: event(x, a),
  "map": lambda x, a: heatmap(x, a)
}

def main(command, args):
  print(f"Loading data...")
  with open(CSV_FILE, mode='r', encoding='utf-8') as file:
    dict_reader = csv.DictReader(file)
    COMMANDS[command](dict_reader, args)

if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2:])