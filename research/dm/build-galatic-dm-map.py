import numpy as np
import pickle
from astropy_healpix import HEALPix
from mwprop.nemod.NE2025 import ne2025
from astropy.coordinates import SkyCoord
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

NSIDE = 16
hp = HEALPix(nside=NSIDE, order='ring', frame='galactic')

def compute_pix(pix):
  coord = hp.healpix_to_skycoord(pix)
  l = coord.l.degree
  b = coord.b.degree

  Dk, Dv, Du, Dd = ne2025(ldeg=l, bdeg=b, dmd=1e6, ndir=1, classic=False)
  return pix, float(Dv['DM'])

def _init_pool():
  # avoids repeated imports overhead in workers (optional but cleaner)
  pass

if __name__ == "__main__":

  dm_map = np.zeros(hp.npix)

  with Pool(cpu_count()) as pool:
    results = list(
      tqdm(
        pool.imap(compute_pix, range(hp.npix)),
        total=hp.npix,
        ncols=80,
        unit="pix",
        desc="Building HEALPix DM map"
      )
    )

  for pix, val in results:
    dm_map[pix] = val

  with open("dm_mw_healpix.pkl", "wb") as f:
    pickle.dump(dm_map, f)