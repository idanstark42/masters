import numpy as np
from scipy.stats import binned_statistic_2d
from tqdm import tqdm
from astropy.coordinates import SkyCoord
import astropy.units as u

from galaxies_decorator import RegladeGalaxyProvider
from utils import RIGHT_ASCENSION_FIELD, DECLENATION_FIELD

class BaseExtractor:
    def _get_raw_data(self, command_instance, asc, dec, area, distance_range):
        """Must be implemented by subclasses to return: (ras, decs, values, statistic_string)"""
        raise NotImplementedError

    def extract_cartesian(self, command_instance, asc, dec, area, distance_range, ra_edges, dec_edges, graining="grid", normalize=False):
        ras, decs, values, statistic = self._get_data(command_instance, asc, dec, area, distance_range, normalize)
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

    def extract_polar(self, command_instance, asc, dec, area, distance_range, hp, graining="grid", sigma=1.0, normalize=False):
        ras, decs, values, statistic = self._get_data(command_instance, asc, dec, area, distance_range, normalize)

        if len(ras) == 0:
            return np.full(hp.npix, np.nan)

        if graining == 'grid':
            coords = SkyCoord(ra=np.array(ras) * u.deg, dec=np.array(decs) * u.deg)
            pixel_indices = hp.skycoord_to_healpix(coords)
            
            healpix_map = np.zeros(hp.npix, dtype=np.float64)
            counts = np.bincount(pixel_indices, minlength=hp.npix)

            if statistic == 'count':
                healpix_map = counts.astype(np.float64)
                healpix_map[counts == 0] = np.nan
                
            elif statistic == 'sum':
                np.add.at(healpix_map, pixel_indices, values)
                healpix_map[counts == 0] = np.nan
                
            elif statistic == 'mean':
                sum_map = np.zeros(hp.npix, dtype=np.float64)
                np.add.at(sum_map, pixel_indices, values)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    healpix_map = sum_map / counts
                    
            return healpix_map

        elif graining == 'gaussian':
            # Get the RA and Dec of the center of every HEALPix pixel
            pixel_indices = np.arange(hp.npix)
            lon, lat = hp.healpix_to_lonlat(pixel_indices)
            ra_centers = lon.deg
            dec_centers = lat.deg

            res = np.zeros(hp.npix, dtype=np.float64)
            weight_sum = np.zeros(hp.npix, dtype=np.float64)

            if values is None:
                values = np.ones_like(ras)

            # Apply the same Gaussian weight logic to the pixel centers
            for ra, dec, val in tqdm(zip(ras, decs, values), total=len(ras), desc=f"HEALPix Gaussian ({statistic})", unit="obj"):
                dx = (ra_centers - ra)
                dx = (dx + 180) % 360 - 180
                dx = dx * np.cos(np.radians(dec_centers))
                dy = dec_centers - dec
                
                gauss = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

                if statistic in ['count', 'sum']:
                    res += val * gauss
                elif statistic == 'mean':
                    res += val * gauss
                
                weight_sum += gauss

            valid = weight_sum > 1e-4
            
            out = np.full(hp.npix, np.nan)
            if statistic == 'mean':
                out[valid] = res[valid] / weight_sum[valid]
            else:
                out[valid] = res[valid]
                
            return out

    def _get_data (self, command_instance, asc, dec, area, distance_range, normalize=False):
        ras, decs, values, statistic = self._get_raw_data(command_instance, asc, dec, area, distance_range)
        # if normalize is True, normalize the values to the range [-1, 1] with the average value as the center
        if normalize and len(values) > 0:
            avg_value = np.mean(values)
            max_dev = max(np.max(values - avg_value), np.max(avg_value - values))
            if max_dev > 0:
                values = (values - avg_value) / max_dev
        return ras, decs, values, statistic

# ---------------------------------------------------------
# DOMAIN EXTRACTORS
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# SPECIFIC VARIABLE EXTRACTORS
# ---------------------------------------------------------

class GalaxyMassDensityExtractor(GalaxyExtractor):
    def _get_raw_data(self, command_instance, asc, dec, area, distance_range):
        df = self.galaxies_in_area(asc, dec, area, distance_range)
        if df.empty or 'M' not in df.columns: 
            return [], [], None, 'sum'
        return df['RA'].values, df['Dec'].values, df['M'].values, 'sum'

class GalaxyNumberDensityExtractor(GalaxyExtractor):
    def _get_raw_data(self, command_instance, asc, dec, area, distance_range):
        df = self.galaxies_in_area(asc, dec, area, distance_range)
        if df.empty: 
            return [], [], None, 'count'
        return df['RA'].values, df['Dec'].values, None, 'count'

class GalaxyAvgMassExtractor(GalaxyExtractor):
    def _get_raw_data(self, command_instance, asc, dec, area, distance_range):
        df = self.galaxies_in_area(asc, dec, area, distance_range)
        if df.empty or 'M' not in df.columns: 
            return [], [], None, 'mean'
        return df['RA'].values, df['Dec'].values, df['M'].values, 'mean'

class FrbDensityExtractor(FRBExtractor):
    def _get_raw_data(self, command_instance, asc, dec, area, distance_range):
        ras, decs, _ = self.frbs_in_area(command_instance, asc, dec, area)
        return ras, decs, None, 'count'

class FrbAvgDmExtractor(FRBExtractor):
    def _get_raw_data(self, command_instance, asc, dec, area, distance_range):
        ras, decs, dms = self.frbs_in_area(command_instance, asc, dec, area)
        return ras, decs, dms, 'mean'


VARIABLE_TYPES = {
    "galaxy.mass-density": { "name": "Galaxy Mass Density", "extractor": GalaxyMassDensityExtractor },
    "galaxy.number-density": { "name": "Galaxy Number Density", "extractor": GalaxyNumberDensityExtractor },
    "galaxy.avg-mass": { "name": "Galaxy Average Mass", "extractor": GalaxyAvgMassExtractor },
    "frb.density": { "name": "FRB Density", "extractor": FrbDensityExtractor },
    "frb.avg-dm": { "name": "FRB Average DM", "extractor": FrbAvgDmExtractor }
}