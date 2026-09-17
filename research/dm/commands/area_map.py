import numpy as np
import matplotlib.pyplot as plt
from commands.command import Command
from utils import parse_area_args, parse_range, RIGHT_ASCENSION_FIELD, DECLENATION_FIELD
from galaxies_decorator import GalaxiesProvider
from baryonic_matter_decorator import BaryonicMatterProvider

BACKGROUND_DOT_SIZE = 20  # Default size for background points (galaxies and baryons)

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
        print(f"Found {len(ras)} events in the specified area.")

        galaxies_provider = GalaxiesProvider()
        galaxies_df = galaxies_provider.get_galaxies_in_area(asc, dec, area, distance_range)
        galaxies_sample = galaxies_df.sample(frac=10000/len(galaxies_df)) if len(galaxies_df) > 10000 else galaxies_df
        galaxies_sizes = galaxies_sample['M'] / np.max(galaxies_sample['M']) * BACKGROUND_DOT_SIZE if 'M' in galaxies_sample.columns else 20
        plt.scatter(galaxies_sample['RA'], galaxies_sample['Dec'], s=galaxies_sizes, alpha=0.5, c='black', label='Galaxies')

        baryonic_matter_provider = BaryonicMatterProvider()
        baryons_df = baryonic_matter_provider.get_sources_in_area(asc=asc, dec=dec, area=area)
        baryon_sample = baryons_df.sample(frac=10000/len(baryons_df)) if len(baryons_df) > 10000 else baryons_df
        baryon_sizes = baryon_sample['Total_Flux'] / np.max(baryon_sample['Total_Flux']) * BACKGROUND_DOT_SIZE if 'Total_Flux' in baryon_sample.columns else 20

        print(f"Successfully loaded {len(galaxies_sample)} galaxies and {len(baryon_sample)} baryonic sources. Generating plot...")
        
        plt.figure(figsize=(10, 8))
        
        if not baryon_sample.empty:
            plt.scatter(baryon_sample['RA'], baryon_sample['Dec'], s=baryon_sizes, alpha=0.5, c='red', label='Baryonic Sources')
        
        if len(dm_excs) > 0:
            plt.scatter(
                ras, decs, 
                s=np.array(dm_excs) / np.max(dm_excs) * 150, 
                alpha=0.5,
                c='blue',
                label='FRBs (Size = DM)'
            )
            
        plt.legend()
        plt.xlim(asc - area, asc + area)
        plt.ylim(dec - area, dec + area)
        plt.xlabel('Right Ascension (deg)')
        plt.ylabel('Declination (deg)')
        plt.title(f'FRBs, Galaxies & Baryonic Matter\nCentered at RA={asc}, Dec={dec}')
        plt.grid(True, alpha=0.3)
        
        filename = f'figures/area_map_heatmap_RA{asc}_Dec{dec}_Area{area}_Range{distance_range[0]}_{distance_range[1]}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
        plt.show()