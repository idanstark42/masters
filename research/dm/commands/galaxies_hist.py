import numpy as np
import matplotlib.pyplot as plt
from commands.command import Command
from utils import parse_range
from galaxies_decorator import GalaxiesProvider

class GalaxiesMapCommand(Command):
    def run(self, args):
        print(f"Scanning the GLADE+ catalog to map galaxy distances")
        
        galaxies_provider = GalaxiesProvider()
        galaxies_df = galaxies_provider.get_galaxies_in_area(asc=180.0, dec=0.0, area=180.0)
        
        if galaxies_df.empty:
            print("No galaxies were successfully parsed or found within the range!")
            return
            
        sample = galaxies_df.sample(frac=0.01) if len(galaxies_df) > 1000 else galaxies_df
        dls = sample['D_L']
        
        bins = np.logspace(np.log10(np.min(dls)), np.log10(np.max(dls)), 61)
        
        plt.figure(figsize=(12, 6))
        plt.hist(dls, bins=bins)
        plt.xscale('log')
        plt.xlabel('Luminosity Distance (Mpc')
        plt.ylabel('Number of Galaxies')
        plt.title(f'Galaxies Luminosity Distance Histogram')
        plt.savefig('dl_histogram.png', dpi=300)
        plt.show()