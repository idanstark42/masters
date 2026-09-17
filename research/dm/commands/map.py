import numpy as np
import matplotlib.pyplot as plt
from commands.command import Command
from utils import parse_area_args, parse_range, RIGHT_ASCENSION_FIELD, DECLENATION_FIELD
from galaxies_decorator import GalaxiesProvider
from baryonic_matter_decorator import BaryonicMatterProvider

BACKGROUND_DOT_SIZE = 20  # Default size for background points (galaxies and baryons)
EVENT_DOT_SIZE = 100

class AreaMapCommand(Command):
    def run(self, args):
        asc, dec, area, radius = parse_area_args(args)
        distance_range = parse_range(args)

        plt.figure(figsize=(10, 8))

        ras, decs, dm_excs = [], [], []
        if 'events' in args:
            for ev in self.iterate_events():
                if abs(float(ev[RIGHT_ASCENSION_FIELD]) - asc) <= area and abs(float(ev[DECLENATION_FIELD]) - dec) <= area:
                    ras.append(float(ev[RIGHT_ASCENSION_FIELD]))
                    decs.append(float(ev[DECLENATION_FIELD]))
                    dm_excs.append(float(ev["dm_exc"]))
            if len(ras) > 0:
                print(f"Found {len(ras)} events in the specified area.")
                plt.scatter(
                    ras, decs, 
                    s=np.array(dm_excs) / np.max(dm_excs) * EVENT_DOT_SIZE if len(dm_excs) > 0 else 20, 
                    alpha=0.5,
                    label='FRBs (Size = DM)'
                )
            else:
                print("No events found in the specified area.")

        if 'galaxies' in args:
            galaxies_provider = GalaxiesProvider()
            galaxies_df = galaxies_provider.get_galaxies_in_area(asc, dec, area, distance_range)
            galaxies_sample = galaxies_df.sample(frac=10000/len(galaxies_df)) if len(galaxies_df) > 10000 else galaxies_df
            galaxies_sizes = galaxies_sample['M'] / np.max(galaxies_sample['M']) * BACKGROUND_DOT_SIZE if 'M' in galaxies_sample.columns else 20
            if not galaxies_sample.empty:
                plt.scatter(galaxies_sample['RA'], galaxies_sample['Dec'], s=galaxies_sizes, alpha=0.5, c='black', label='Galaxies')
                print(f"Successfully loaded {len(galaxies_sample)} galaxies.")
            else:
                print("No galaxies found in the specified area.")

        if 'baryons' in args:
            baryonic_matter_provider = BaryonicMatterProvider()
            baryons_df = baryonic_matter_provider.get_sources_in_area(asc=asc, dec=dec, area=area)
            baryon_sample = baryons_df.sample(frac=10000/len(baryons_df)) if len(baryons_df) > 10000 else baryons_df
            baryon_sizes = baryon_sample['Total_Flux'] / np.max(baryon_sample['Total_Flux']) * BACKGROUND_DOT_SIZE if 'Total_Flux' in baryon_sample.columns else 20
            if not baryon_sample.empty:
                plt.scatter(baryon_sample['RA'], baryon_sample['Dec'], s=baryon_sizes, alpha=0.5, c='red', label='Baryonic Sources')
                print(f"Successfully loaded {len(baryon_sample)} baryonic sources.")
            else:
                print("No baryonic sources found in the specified area.")

        plt.legend()
        plt.xlim(np.max([asc - area, 0]), np.min([asc + area, 360]))
        plt.ylim(np.max([dec - area, -90]), np.min([dec + area, 90]))
        plt.xlabel('Right Ascension (deg)')
        plt.ylabel('Declination (deg)')
        plt.title(f'FRBs, Galaxies & Baryonic Matter\nCentered at RA={asc}, Dec={dec}')
        plt.grid(True, alpha=0.3)
        
        filename = f'figures/area_map_heatmap_RA{asc}_Dec{dec}_Area{area}_Range{distance_range[0]}_{distance_range[1]}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
        plt.show()