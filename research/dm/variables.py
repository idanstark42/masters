import numpy as np
from scipy.stats import binned_statistic_2d
from tqdm import tqdm
from galaxies_decorator import RegladeGalaxyProvider
from utils import RIGHT_ASCENSION_FIELD, DECLENATION_FIELD

class BaseExtractor:
    def _process_data(self, ras, decs, values, statistic, ra_edges, dec_edges, graining='grid', sigma=1.0):
        # Handle empty datasets by returning an array of NaNs
        if len(ras) == 0:
            return np.full((len(ra_edges)-1, len(dec_edges)-1), np.nan)

        if graining == 'grid':
            # Always calculate count to identify and mask zero-occupancy bins
            counts, _, _, _ = binned_statistic_2d(ras, decs, values=None, statistic='count', bins=[ra_edges, dec_edges])
            
            if statistic == 'count':
                res = counts
            else:
                res, _, _, _ = binned_statistic_2d(ras, decs, values=values, statistic=statistic, bins=[ra_edges, dec_edges])
            
            # Mask out areas with 0 galaxies/FRBs to ignore them in correlations and maps
            res = res.astype(np.float64)
            res[counts == 0] = np.nan
            return res

        elif graining == 'gaussian':
            # Create grid centers to sample the Gaussians
            ra_centers = (ra_edges[:-1] + ra_edges[1:]) / 2
            dec_centers = (dec_edges[:-1] + dec_edges[1:]) / 2
            
            # indexing='ij' ensures output shape matches the (RA, Dec) binning format
            X, Y = np.meshgrid(ra_centers, dec_centers, indexing='ij')

            res = np.zeros_like(X, dtype=np.float64)
            weight_sum = np.zeros_like(X, dtype=np.float64)

            if values is None:
                values = np.ones_like(ras)

            # Wrap the zip iterator with tqdm
            for ra, dec, val in tqdm(zip(ras, decs, values), total=len(ras), desc=f"Gaussian smoothing ({statistic})", unit="obj"):
                dx = (X - ra)
                dx = (dx + 180) % 360 - 180
                dx = dx * np.cos(np.radians(Y))
                dy = Y - dec
                
                gauss = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

                if statistic in ['count', 'sum']:
                    res += val * gauss
                elif statistic == 'mean':
                    res += val * gauss
                
                weight_sum += gauss

            # Identify areas with effectively zero Gaussian overlap
            valid = weight_sum > 1e-4
            
            out = np.full_like(X, np.nan)
            if statistic == 'mean':
                out[valid] = res[valid] / weight_sum[valid]
            else:
                out[valid] = res[valid]
                
            return out

class GalaxyExtractor(BaseExtractor):
    def galaxies_in_area(self, asc, dec, area, distance_range):
        return RegladeGalaxyProvider().get_galaxies_in_area(asc, dec, area, distance_range)

class FRBExtractor(BaseExtractor):
    def frbs_in_area(self, command_instance, asc, dec, area):
        ras, decs, dms = [], [], []
        for ev in command_instance.iterate_events():
            ra, d = float(ev[RIGHT_ASCENSION_FIELD]), float(ev[DECLENATION_FIELD])
            if abs(ra - asc) <= area and abs(d - dec) <= area:
                ras.append(ra)
                decs.append(d)
                dms.append(float(ev["dm_exc"]))
        return ras, decs, dms


class GalaxyMassDensityExtractor(GalaxyExtractor):
    def extract(self, command_instance, asc, dec, area, distance_range, ra_edges, dec_edges, graining="grid"):
        df = self.galaxies_in_area(asc, dec, area, distance_range)
        if df.empty or 'M' not in df.columns: 
            return self._process_data([], [], None, 'sum', ra_edges, dec_edges, graining)
        return self._process_data(df['RA'].values, df['Dec'].values, df['M'].values, 'sum', ra_edges, dec_edges, graining)

class GalaxyNumberDensityExtractor(GalaxyExtractor):
    def extract(self, command_instance, asc, dec, area, distance_range, ra_edges, dec_edges, graining="grid"):
        df = self.galaxies_in_area(asc, dec, area, distance_range)
        if df.empty: 
            return self._process_data([], [], None, 'count', ra_edges, dec_edges, graining)
        return self._process_data(df['RA'].values, df['Dec'].values, None, 'count', ra_edges, dec_edges, graining)

class GalaxyAvgMassExtractor(GalaxyExtractor):
    def extract(self, command_instance, asc, dec, area, distance_range, ra_edges, dec_edges, graining="grid"):
        df = self.galaxies_in_area(asc, dec, area, distance_range)
        if df.empty or 'M' not in df.columns: 
            return self._process_data([], [], None, 'mean', ra_edges, dec_edges, graining)
        return self._process_data(df['RA'].values, df['Dec'].values, df['M'].values, 'mean', ra_edges, dec_edges, graining)

class FrbDensityExtractor(FRBExtractor):
    def extract(self, command_instance, asc, dec, area, distance_range, ra_edges, dec_edges, graining="grid"):
        ras, decs, _ = self.frbs_in_area(command_instance, asc, dec, area)
        return self._process_data(ras, decs, None, 'count', ra_edges, dec_edges, graining)

class FrbAvgDmExtractor(FRBExtractor):
    def extract(self, command_instance, asc, dec, area, distance_range, ra_edges, dec_edges, graining="grid"):
        ras, decs, dms = self.frbs_in_area(command_instance, asc, dec, area)
        return self._process_data(ras, decs, dms, 'mean', ra_edges, dec_edges, graining)


VARIABLE_TYPES = {
    "galaxy.mass-density": { "name": "Galaxy Mass Density", "extractor": GalaxyMassDensityExtractor },
    "galaxy.number-density": { "name": "Galaxy Number Density", "extractor": GalaxyNumberDensityExtractor },
    "galaxy.avg-mass": { "name": "Galaxy Average Mass", "extractor": GalaxyAvgMassExtractor },
    "frb.density": { "name": "FRB Density", "extractor": FrbDensityExtractor },
    "frb.avg-dm": { "name": "FRB Average DM", "extractor": FrbAvgDmExtractor }
}