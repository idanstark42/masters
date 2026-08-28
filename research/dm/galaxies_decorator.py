import pandas as pd
import numpy as np
import tqdm
from utils import glade_columns, GLADE_FILE

class GalaxiesProvider:
    def __init__(self, glade_file=GLADE_FILE):
        self.glade_file = glade_file

    def get_galaxies_in_area(self, asc, dec, area, distance_range=(0, float('inf'))):
        """Returns a DataFrame of galaxies within the bounding box and distance range safely."""
        print(f"\nSearching GLADE catalog for galaxies near RA={asc}, Dec={dec} within distance {distance_range} Mpc...")
        
        filtered_chunks = []
        chunk_size = 100000
        # Estimate total chunks for ~22.5M rows
        estimated_total_chunks = 22500000 // chunk_size

        chunk_iterator = pd.read_csv(
            self.glade_file, 
            sep=r'\s+', 
            header=None, 
            usecols=[8, 9, 32, 35],
            names=['RA', 'Dec', 'D_L', 'M'],
            na_values=['null'], 
            low_memory=False,
            chunksize=chunk_size
        )
    
        min_dist, max_dist = distance_range

        for chunk in tqdm.tqdm(chunk_iterator, total=estimated_total_chunks, desc="Scanning GLADE"):
            # Force numeric types to prevent silent failures
            chunk['RA'] = pd.to_numeric(chunk['RA'], errors='coerce')
            chunk['Dec'] = pd.to_numeric(chunk['Dec'], errors='coerce')
            chunk['D_L'] = pd.to_numeric(chunk['D_L'], errors='coerce')

            match = chunk[
                chunk['RA'].between(asc - area, asc + area) & 
                chunk['Dec'].between(dec - area, dec + area) &
                chunk['D_L'].between(min_dist, max_dist)
            ]
            if not match.empty:
                filtered_chunks.append(match)

        if len(filtered_chunks) > 0:
            return pd.concat(filtered_chunks, ignore_index=True)
        else:
            return pd.DataFrame(columns=['RA', 'Dec', 'D_L'])