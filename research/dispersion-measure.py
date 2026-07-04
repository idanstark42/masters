import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

CSV_FILE = './canfar.net_storage_vault_file_AstroDataCitationDOI_CISTI.CANFAR_25.0066_data_table_chimefrbcat2.csv'

DISPERSION_MEASURE_FIELD = 'dm_fitb'
RIGHT_ASCENSION_FIELD = 'ra'
DECLENATION_FIELD = 'dec'

DEFAULT_RA_RES = 4
DEFAULT_DEC_RES = 4

def keys(dict_reader, args):
  print(dict_reader.fieldnames)

def event(dict_reader, args):
  index = args[0]
  rows = list(dict_reader)
  print(rows[int(index)])

def heatmap(dict_reader, args):
  field = args[0] if len(args) > 0 else DISPERSION_MEASURE_FIELD
  right_ascention_res = DEFAULT_RA_RES
  declenation_res = DEFAULT_DEC_RES

  if len(args) > 1:
    right_ascention_res, declenation_res = map(int, args[1].split(':'))

  values = []
  ras = []
  decs = []

  for row in dict_reader:
    try:
      ras.append(float(row[RIGHT_ASCENSION_FIELD]))
      decs.append(float(row[DECLENATION_FIELD]))
      values.append(float(row[field]))
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
  plt.colorbar(label=field)
  plt.xlabel('Right Ascension (deg)')
  plt.ylabel('Declenation (deg)')
  plt.title(f'Average {field}')
  plt.show()

COMMANDS = {
  "keys": lambda x, a: keys(x, a),
  "event": lambda x, a: event(x, a),
  "map": lambda x, a: heatmap(x, a)
}

def main(command, args):
  with open(CSV_FILE, mode='r', encoding='utf-8') as file:
    dict_reader = csv.DictReader(file)
    COMMANDS[command](dict_reader, args)

if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2:])