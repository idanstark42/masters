import os
import hashlib
from functools import wraps
import pandas as pd
import numpy as np
import tqdm
from astropy.table import Table

class BaryonicMatterProvider:
    def __init__(self, processed_fits_path="data/erass_fluxes.fits"):
        self.processed_fits_path = processed_fits_path
        print(f"Loading processed eRASS data from {self.processed_fits_path}...")
        fits_table = Table.read(self.processed_fits_path, format='fits')
        self.df = fits_table.to_pandas()

    def get_sources_in_area(self, asc, dec, area, min_flux=0.0):
        """
        Returns a DataFrame of eRASS X-ray sources within the bounding box 
        and above a minimum total flux threshold safely.
        """
        print(f"Filtering eRASS sources near RA={asc}, Dec={dec} within area ±{area}°...")
        
        # Ensure correct numeric types
        self.df['RA'] = pd.to_numeric(self.df['RA'], errors='coerce')
        self.df['Dec'] = pd.to_numeric(self.df['Dec'], errors='coerce')
        self.df['Total_Flux'] = pd.to_numeric(self.df['Total_Flux'], errors='coerce')

        match = self.df[
            self.df['RA'].between(asc - area, asc + area) & 
            self.df['Dec'].between(dec - area, dec + area) &
            (self.df['Total_Flux'] >= min_flux)
        ]
        
        return match.reset_index(drop=True)