import numpy as np
import matplotlib.pyplot as plt
from commands.command import Command
from utils import parse_area_args, parse_range, RIGHT_ASCENSION_FIELD, DECLENATION_FIELD
from galaxies_decorator import GalaxiesProvider

class AreaMapCommand(Command):
    def run(self, args):
        asc, dec, area, radius = parse_area_args(args)
        distance_range = parse_range(args)
        if asc is None or dec is None:
            print("No position specified. Use 'pos=RA,Dec'.")
            return

        ras, decs, dm_excs = [], [], []
        for ev in self.iterate_events():
            if abs(float(ev[RIGHT_ASCENSION_FIELD]) - asc) <= area and abs(float(ev[DECLENATION_FIELD]) - dec) <= area:
                ras.append(float(ev[RIGHT_ASCENSION_FIELD]))
                decs.append(float(ev[DECLENATION_FIELD]))
                dm_excs.append(float(ev["dm_exc"]))

        galaxies_provider = GalaxiesProvider()
        galaxies_df = galaxies_provider.get_galaxies_in_area(asc, dec, area, distance_range)
        
        sample = galaxies_df.sample(frac=0.5) if len(galaxies_df) > 10000 else galaxies_df
        sizes = sample['M']
        print(f"Successfully loaded {len(sample)} galaxies (50% sample). Generating plot...")
        
        plt.figure(figsize=(10, 8))
        plt.scatter(sample['RA'], sample['Dec'], s=sizes, alpha=0.5, c='black')
        
        if len(dm_excs) > 0:
            plt.scatter(
                ras, decs, 
                s=np.array(dm_excs) / np.max(dm_excs) * 150, 
                alpha=0.5, 
                label='Events (Size = DM)'
            )
            
        plt.legend()
        plt.xlim(asc - area, asc + area)
        plt.ylim(dec - area, dec + area)
        plt.xlabel('Right Ascension (deg)')
        plt.ylabel('Declination (deg)')
        plt.title(f'Event Locations vs Galaxy Density\nCentered at RA={asc}, Dec={dec}')
        plt.grid(True, alpha=0.3)
        
        filename = f'figures/area_map_heatmap_RA{asc}_Dec{dec}_Area{area}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
        plt.show()