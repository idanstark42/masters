import os
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from commands.command import Command
from utils import parse_area_args, parse_range, RIGHT_ASCENSION_FIELD, DECLENATION_FIELD
from galaxies_decorator import GalaxiesProvider
from astropy.coordinates import SkyCoord
import astropy.units as u

class DmVsDensityCommand(Command):
    def run(self, args):
        asc, dec, area, radius = parse_area_args(args)
        distance_range = parse_range(args)
        if asc is None or dec is None:
            print("No position specified. Use 'pos=RA,Dec'.")
            return

        data_filename = f'data/dm_vs_density_data_RA{asc}_Dec{dec}_Area{area}_Rad{radius}.csv'

        if os.path.exists(data_filename):
            print(f"Found cached data! Loading directly from {data_filename}...")
            cached_df = pd.read_csv(data_filename)
            densities = cached_df['Density'].values
            event_dms = cached_df['DM_Excess'].values
            
        else:
            print(f"Scanning for events in area RA={asc}, Dec={dec}...")
            event_ras, event_decs, event_dms = [], [], []

            for ev in self.iterate_events():
                ra = float(ev[RIGHT_ASCENSION_FIELD])
                decl = float(ev[DECLENATION_FIELD])
                if abs(ra - asc) <= area and abs(decl - dec) <= area:
                    event_ras.append(ra)
                    event_decs.append(decl)
                    event_dms.append(float(ev["dm_exc"]))
                    
            if not event_ras:
                print("No events found in this area.")
                return

            print(f"Found {len(event_ras)} events. Fetching GLADE galaxies...")

            galaxies_provider = GalaxiesProvider()
            galaxies_df = galaxies_provider.get_galaxies_in_area(asc, dec, area + radius, distance_range)

            if galaxies_df.empty:
                print("No galaxies found in this area to calculate density.")
                return

            print(f"Calculating galaxy density within {radius}° of each event...")
            
            event_coords = SkyCoord(ra=event_ras*u.degree, dec=event_decs*u.degree)
            galaxy_coords = SkyCoord(ra=galaxies_df['RA'].values*u.degree, dec=galaxies_df['Dec'].values*u.degree)

            densities = []
            circle_area = np.pi * (radius ** 2)

            for i in range(len(event_coords)):
                seps = event_coords[i].separation(galaxy_coords)
                count = np.sum(seps.degree <= radius)
                density = count / circle_area
                densities.append(density)

            print(f"Saving computed data to {data_filename}...")
            output_df = pd.DataFrame({
                'RA': event_ras,
                'Dec': event_decs,
                'DM_Excess': event_dms,
                'Density': densities
            })
            os.makedirs('figures', exist_ok=True)
            output_df.to_csv(data_filename, index=False)

        print("Generating plot...")
        plt.figure(figsize=(10, 6))
        
        density_mean = np.mean(densities)
        dm_mean = np.mean(event_dms)
        density_std = np.std(densities)
        dm_std = np.std(event_dms)
        
        filtered_densities, filtered_event_dms = [], []
        for i in range(len(densities)):
            density = densities[i]
            dm = event_dms[i]
            if abs(density - density_mean) < 3 * density_std and abs(dm - dm_mean) < 3 * dm_std:
                filtered_densities.append(density)
                filtered_event_dms.append(dm)

        plt.scatter(filtered_densities, filtered_event_dms, alpha=0.7, c='purple', edgecolor='k')
        
        if len(filtered_densities) > 1:
            z = np.polyfit(filtered_densities, filtered_event_dms, 1)
            p = np.poly1d(z)
            r = np.corrcoef(filtered_event_dms, p(filtered_densities))[0, 1]
            r_squared = r**2
            plt.plot(filtered_densities, p(filtered_densities), "r--", alpha=0.8, label=f'Linear Trend (R²={r_squared:.2f})')

        plt.xlabel(f'Local Galaxy Density (galaxies / sq degree)\nwithin {radius}° radius')
        plt.ylabel('Dispersion Measure Excess (pc cm$^{-3}$)')
        plt.title(f'DM Excess vs Local Galaxy Density\nCentered at RA={asc}, Dec={dec}')
        plt.legend()
        
        plot_filename = f'figures/dm_vs_density_plot_RA{asc}_Dec{dec}_Area{area}_Rad{radius}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {plot_filename}")
        plt.show()