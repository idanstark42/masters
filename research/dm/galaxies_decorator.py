import pandas as pd
import numpy as np
import tqdm
from astropy.io import fits

# Assuming you have or will add GLADE_FILE and REGLADE_FILE to your utils
from utils import GLADE_FILE, REGLADE_FILE

class GalaxyProvider:
    def __init__(self, catalog_file):
        self.catalog_file = catalog_file

    def _get_chunk_iterator(self, chunk_size):
        """Must return a generator yielding DataFrame chunks and the total number of chunks."""
        raise NotImplementedError("Subclasses must implement _get_chunk_iterator.")

    def _get_column_names(self):
        """Must return a dict mapping standard column names to actual catalog column names."""
        raise NotImplementedError("Subclasses must implement _get_column_names.")

    def _format_output(self, df):
        """Optional hook to standardize output columns. Subclasses can override."""
        return df

    def get_galaxies_in_area(self, asc, dec, area, distance_range=(0, float('inf'))):
        """Returns a DataFrame of galaxies within the bounding box and distance range safely."""
        cols = self._get_column_names()
        ra_col = cols['ra']
        dec_col = cols['dec']
        dist_col = cols['dist']
        desc = cols['desc']
        
        print(f"\n{desc} for galaxies near RA={asc}, Dec={dec} within distance {distance_range} Mpc...")
        
        filtered_chunks = []
        chunk_size = 100000
        
        chunk_iterator, total_chunks = self._get_chunk_iterator(chunk_size)
        min_dist, max_dist = distance_range

        for chunk in tqdm.tqdm(chunk_iterator, total=total_chunks, desc=desc, ncols=80):
            # Force numeric types to prevent silent failures
            chunk[ra_col] = pd.to_numeric(chunk[ra_col], errors='coerce')
            chunk[dec_col] = pd.to_numeric(chunk[dec_col], errors='coerce')
            chunk[dist_col] = pd.to_numeric(chunk[dist_col], errors='coerce')

            match = chunk[
                chunk[ra_col].between(asc - area, asc + area) & 
                chunk[dec_col].between(dec - area, dec + area) &
                chunk[dist_col].between(min_dist, max_dist)
            ]
            if not match.empty:
                filtered_chunks.append(match)

        if len(filtered_chunks) > 0:
            result = pd.concat(filtered_chunks, ignore_index=True)
            return self._format_output(result)
        else:
            # Return standardized empty DataFrame
            return pd.DataFrame(columns=['RA', 'Dec', 'D_L', 'M'])


class GladeGalaxyProvider(GalaxyProvider):
    def __init__(self, catalog_file=GLADE_FILE):
        super().__init__(catalog_file)

    def _get_chunk_iterator(self, chunk_size):
        iterator = pd.read_csv(
            self.catalog_file, 
            sep=r'\s+', 
            header=None, 
            usecols=[8, 9, 32, 35],
            names=['RA', 'Dec', 'D_L', 'M'],
            na_values=['null'], 
            low_memory=False,
            chunksize=chunk_size
        )
        
        estimated_total_chunks = 22500000 // chunk_size
        return iterator, estimated_total_chunks

    def _get_column_names(self):
        return {'ra': 'RA', 'dec': 'Dec', 'dist': 'D_L', 'desc': 'Scanning GLADE catalog'}


class RegladeGalaxyProvider(GalaxyProvider):
    def __init__(self, catalog_file=REGLADE_FILE):
        super().__init__(catalog_file)

    def _get_chunk_iterator(self, chunk_size):        
        # Read just the header first to find the exact number of rows
        with fits.open(self.catalog_file) as hdul:
            total_rows = hdul[1].header.get('NAXIS2', 24000000)
            
        total_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)

        def generator():
            # memmap=True is critical here; it prevents loading the whole FITS array into RAM
            with fits.open(self.catalog_file, memmap=True) as hdul:
                data = hdul[1].data

                for start in range(0, total_rows, chunk_size):
                    end = min(start + chunk_size, total_rows)
                    
                    # FITS arrays are typically Big-Endian, which Pandas rejects. 
                    # Casting directly to np.float64 fixes endianness and standardizes the dtype.
                    chunk_df = pd.DataFrame({
                        'ra': np.array(data['gal_ra'][start:end], dtype=np.float64),
                        'dec': np.array(data['gal_dec'][start:end], dtype=np.float64),
                        'D': np.array(data['D'][start:end], dtype=np.float64),
                        'logM': np.array(data['logM'][start:end], dtype=np.float64)
                    })
                    yield chunk_df

        return generator(), total_chunks

    def _get_column_names(self):
        return {'ra': 'ra', 'dec': 'dec', 'dist': 'D', 'desc': 'Scanning REGLADE catalog'}
        
    def _format_output(self, df):
        """
        Renames the REGLADE columns to match the standard GLADE format 
        so downstream code doesn't have to change.
        """
        # change logM to M by running 10**logM to get the actual stellar mass
        if 'logM' in df.columns:
            df['M'] = np.e ** df['logM']

        return df.rename(columns={
            'ra': 'RA', 
            'dec': 'Dec', 
            'D': 'D_L',    # Final recommended distance mapping
        })
