import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy_healpix import HEALPix
import tqdm
import pickle


CSV_FILE = './canfar.net_storage_vault_file_AstroDataCitationDOI_CISTI.CANFAR_25.0066_data_table_chimefrbcat2.csv'
GALACTIC_DM_MAP_FILE = './dm_mw_healpix.pkl'

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

hp = HEALPix(nside=NSIDE, order='ring', frame='galactic')

def keys(dict_reader, args):
  print(dict_reader.fieldnames)

def event(dict_reader, args):
  index = args[0]
  rows = list(dict_reader)
  print(rows[int(index)])

def add_dm_exc(row, galactic_dm_map):
  dm_obs = float(row[DISPERSION_MEASURE_FIELD] or row[DISPERSION_MEASURE_BACKUP_FIELD])
  galactic_latitude = float(row[GALACTIC_LATITUDE_FIELD])
  galactic_longitude = float(row[GALACTIC_LONGITUDE_FIELD])
  coord_gal = SkyCoord(
    l=galactic_longitude*u.degree,
    b=galactic_latitude*u.degree,
    frame='galactic'
  )

  # interpolate MW DM from HEALPix map
  pix = hp.skycoord_to_healpix(coord_gal)
  dm_mw = galactic_dm_map[pix]

  row["dm_exc"] = dm_obs - dm_mw
  return row

# runs all quality cuts and returns True if the event passes all cuts
def quality_event (row, galactic_dm_map):
  if row[EXCLUDE_FIELD] == '1':
    return False
  
  if float(row[SUB_NUM_FIELD]) > 0:
    return False

  if float(row[BONSAI_SNR_FIELD]) < SNR_THRESHOLD:
    return False

  if abs(float(row[GALACTIC_LATITUDE_FIELD])) < GALACTIC_LATITUDE_THRESHOLD:
    return False

  row = add_dm_exc(row, galactic_dm_map)

  if row["dm_exc"] <= 0 or np.isnan(row["dm_exc"]) or np.isnan(float(row[DISPERSION_MEASURE_FIELD] or row[DISPERSION_MEASURE_BACKUP_FIELD])):
    return False

  return True

def each_event(dict_reader):
  with open(GALACTIC_DM_MAP_FILE, "rb") as f:
    galactic_dm_map = pickle.load(f)
    repeaters = []
    rows = list(dict_reader)
    valid_rows = 0
    for row in tqdm.tqdm(rows, desc="Processing events", ncols=80, unit="event"):
      if not quality_event(row, galactic_dm_map):
        continue

      repeater_name = row[REPEATER_NAME_FIELD] if REPEATER_NAME_FIELD in row else None
      if repeater_name:
        if repeater_name in repeaters:
          continue
        else:
          repeaters.append(repeater_name)

      valid_rows += 1
      try:
        yield row
      except (ValueError, KeyError):
        pass

  print(f"Processed events: {valid_rows}/{len(rows)}")

def stats(dict_reader, args):
  dm_obs = []
  dm_excs = []

  if len(args) > 0:
    # split args by =
    pairs = [arg.split('=') for arg in args]
    # find the pair with the key "pos"
    pos_pair = next((pair for pair in pairs if pair[0] == "pos"), None)
    if pos_pair:
      # split the value by ,
      asc, dec = map(float, pos_pair[1].split(','))
      area_pair = next((pair for pair in pairs if pair[0] == "area"), None)
      if area_pair:
        area = float(area_pair[1])
      else:
        area = 5

  for event in each_event(dict_reader):
    if 'asc' in locals() and 'dec' in locals():
      if abs(float(event[RIGHT_ASCENSION_FIELD]) - asc) > area or abs(float(event[DECLENATION_FIELD]) - dec) > area:
        continue
    dm_obs.append(float(event[DISPERSION_MEASURE_FIELD] or event[DISPERSION_MEASURE_BACKUP_FIELD]))
    dm_excs.append(float(event["dm_exc"]))
  
  print(f"Number of events: {len(dm_obs)}")
  print(f"Mean DM excess: {np.mean(dm_excs):.2f}")
  print(f"DM excess std: {np.std(dm_excs):.2f}")

def heatmap(dict_reader, args):
  right_ascention_res = DEFAULT_RA_RES
  declenation_res = DEFAULT_DEC_RES
  min_count = MIN_COUNT

  if len(args) > 0:
    right_ascention_res, declenation_res, min_count = map(int, args[0].split(':'))

  values = []
  ras = []
  decs = []

  for event in each_event(dict_reader):
    ras.append(float(event[RIGHT_ASCENSION_FIELD]))
    decs.append(float(event[DECLENATION_FIELD]))
    values.append(float(event[DISPERSION_MEASURE_FIELD] or event[DISPERSION_MEASURE_BACKUP_FIELD]))

  ra_edges = np.linspace(0, 360, right_ascention_res + 1)
  dec_edges = np.linspace(0, 90, declenation_res + 1)

  if 'smooth' in args:
    sums, _, _ = np.histogram2d(decs, ras, bins=[dec_edges, ra_edges], weights=values)
    counts, _, _ = np.histogram2d(decs, ras, bins=[dec_edges, ra_edges])
  else:
    sums, _, _ = np.histogram2d(decs, ras, bins=[dec_edges, ra_edges], weights=values)
    counts, _, _ = np.histogram2d(decs, ras, bins=[dec_edges, ra_edges])

  averages = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > min_count)

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
  plt.savefig(f'figures/heatmap_RAres{right_ascention_res}_Decres{declenation_res}_Mincount{min_count}.png', dpi=300)
  plt.show()

def eventsmap(dict_reader, args):
  ras = []
  decs = []
  dm_excs = []

  for event in each_event(dict_reader):
    ras.append(float(event[RIGHT_ASCENSION_FIELD]))
    decs.append(float(event[DECLENATION_FIELD]))
    dm_excs.append(float(event["dm_exc"]))

  plt.figure(figsize=(12, 6))
  plt.scatter(ras, decs, s=dm_excs / np.max(dm_excs) * 100, alpha=0.5)
  plt.xlim(0, 360)
  plt.ylim(0, 90)
  plt.xlabel('Right Ascension (deg)')
  plt.ylabel('Declenation (deg)')
  plt.title('Event Locations')
  plt.savefig(f'figures/events_map.png', dpi=300)
  plt.show()

def dm_exc_histogram(dict_reader, args):
  print("Generating DM excess histogram...")

  if len(args) > 0:
    # split args by =
    pairs = [arg.split('=') for arg in args]
    # find the pair with the key "pos"
    pos_pair = next((pair for pair in pairs if pair[0] == "pos"), None)
    if pos_pair:
      # split the value by ,
      asc, dec = map(float, pos_pair[1].split(','))
      area_pair = next((pair for pair in pairs if pair[0] == "area"), None)
      if area_pair:
        area = float(area_pair[1])
      else:
        area = 5

  dm_excs = []

  for event in each_event(dict_reader):
    if 'asc' in locals() and 'dec' in locals():
      if abs(float(event[RIGHT_ASCENSION_FIELD]) - asc) > area or abs(float(event[DECLENATION_FIELD]) - dec) > area:
        continue
    dm_excs.append(float(event["dm_exc"]))
  
  bins = np.logspace(np.log10(np.min(dm_excs)), np.log10(np.max(dm_excs)), 61)

  plt.figure(figsize=(12, 6))
  plt.hist(dm_excs, bins=bins, alpha=0.7)
  plt.xscale('log')
  plt.xlabel('Dispersion Measure Excess (pc cm$^{-3}$)')
  plt.ylabel('Number of Events')
  plt.title('Histogram of Dispersion Measure Excess')
  plt.savefig(f'figures/dm_exc_histogram.png', dpi=300)
  plt.show()

def dm_exc_comparative_histogram(dict_reader, args):
  print("Generating comparative DM excess histogram...")

  if len(args) > 0:
    # split args by =
    pairs = [arg.split('=') for arg in args]
    # find the pair with the key "pos"
    pos_pair = next((pair for pair in pairs if pair[0] == "pos"), None)
    if pos_pair:
      # split the value by ,
      asc, dec = map(float, pos_pair[1].split(','))
      area_pair = next((pair for pair in pairs if pair[0] == "area"), None)
      if area_pair:
        area = float(area_pair[1])
      else:
        area = 5

  if 'asc' not in locals() or 'dec' not in locals():
    print("No position specified. Please provide a position using the 'pos' argument in the format 'pos=RA,Dec'.")
    return

  dm_excs = []
  dm_axcs_in_area = []

  for event in each_event(dict_reader):
    if abs(float(event[RIGHT_ASCENSION_FIELD]) - asc) <= area and abs(float(event[DECLENATION_FIELD]) - dec) <= area:
      dm_axcs_in_area.append(float(event["dm_exc"]))
    dm_excs.append(float(event["dm_exc"]))

  bins = np.logspace(np.log10(np.min(dm_excs)), np.log10(np.max(dm_excs)), 61)

  plt.figure(figsize=(12, 6))
  plt.ylabel('Number of Events')
  # dual y axes
  ax1 = plt.gca()
  ax2 = ax1.twinx()
  ax2.set_ylabel('Number of Events in Area')
  ax1.hist(dm_excs, bins=bins, alpha=0.7, label='All Events')
  ax2.hist(dm_axcs_in_area, bins=bins, alpha=0.7, color='orange', label=f'Events in Area around RA={asc}, Dec={dec}')

  plt.xscale('log')
  plt.xlabel('Dispersion Measure Excess (pc cm$^{-3}$)')
  plt.title('Comparative Histogram of Dispersion Measure Excess')
  plt.legend()
  plt.savefig(f'figures/dm_exc_comparative_histogram_RA{asc}_Dec{dec}_Area{area}.png', dpi=300)
  plt.show()

def area_map(dict_reader, args):
  if len(args) > 0:
    # split args by =
    pairs = [arg.split('=') for arg in args]
    # find the pair with the key "pos"
    pos_pair = next((pair for pair in pairs if pair[0] == "pos"), None)
    if pos_pair:
      # split the value by ,
      asc, dec = map(float, pos_pair[1].split(','))
      area_pair = next((pair for pair in pairs if pair[0] == "area"), None)
      if area_pair:
        area = float(area_pair[1])
      else:
        area = 5

  ras = []
  decs = []
  dm_excs = []

  for event in each_event(dict_reader):
    if abs(float(event[RIGHT_ASCENSION_FIELD]) - asc) <= area and abs(float(event[DECLENATION_FIELD]) - dec) <= area:
      ras.append(float(event[RIGHT_ASCENSION_FIELD]))
      decs.append(float(event[DECLENATION_FIELD]))
      dm_excs.append(float(event["dm_exc"]))

  plt.figure(figsize=(12, 6))
  plt.scatter(ras, decs, s=dm_excs / np.max(dm_excs) * 100, alpha=0.5)
  plt.xlim(asc - area, asc + area)
  plt.ylim(dec - area, dec + area)
  plt.xlabel('Right Ascension (deg)')
  plt.ylabel('Declenation (deg)')
  plt.title(f'Event Locations in Area around RA={asc}, Dec={dec}')
  plt.savefig(f'figures/area_map_RA{asc}_Dec{dec}_Area{area}.png', dpi=300)
  plt.show()

COMMANDS = {
  "keys": lambda x, a: keys(x, a),
  "event": lambda x, a: event(x, a),
  "stats": lambda x, a: stats(x, a),
  "heatmap": lambda x, a: heatmap(x, a),
  "eventmap": lambda x, a: eventsmap(x, a),
  "area-map": lambda x, a: area_map(x, a),
  "dm-hist": lambda x, a: dm_exc_histogram(x, a),
  "dm-comparative-hist": lambda x, a: dm_exc_comparative_histogram(x, a)
}

def main(command, args):
  print(f"Loading data...")
  with open(CSV_FILE, mode='r', encoding='utf-8') as file:
    dict_reader = csv.DictReader(file)
    COMMANDS[command](dict_reader, args)

if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2:])