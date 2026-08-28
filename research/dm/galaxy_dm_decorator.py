import pickle
import numpy as np
from utils import *

class GalaxyDMDecorator:
    def __init__(self, events_manager, dm_map_file):
        self.events_manager = events_manager
        with open(dm_map_file, "rb") as f:
            self.galactic_dm_map = pickle.load(f)

    def iterate_events(self):
        valid_rows = 0
        for row in self.events_manager.iterate_events():
            dm_obs_val = row.get(DISPERSION_MEASURE_FIELD) or row.get(DISPERSION_MEASURE_BACKUP_FIELD)
            
            try:
                dm_obs = float(dm_obs_val)
                ra = row[RIGHT_ASCENSION_FIELD]
                dec = row[DECLENATION_FIELD]
                dm_mw = self.galactic_dm_map[f"{ra}_{dec}"]
                
                dm_exc = dm_obs - dm_mw
                row["dm_exc"] = dm_exc

                if dm_exc <= 0 or np.isnan(dm_exc) or np.isnan(dm_obs):
                    continue
                
                valid_rows += 1
                yield row
                
            except (ValueError, KeyError, TypeError):
                continue