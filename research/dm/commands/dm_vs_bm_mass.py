import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from commands.command import Command
from utils import parse_area_args, parse_range, RIGHT_ASCENSION_FIELD, DECLENATION_FIELD
from baryonic_matter_decorator import BaryonicMatterProvider
from astropy.coordinates import SkyCoord
import astropy.units as u

class DmVsBaryonicDensityCommand(Command):
    def run(self, args):
        asc, dec, area, radius = parse_area_args(args)
        distance_range = parse_range(args)
        if asc is None or dec is None:
            print("No position specified. Use 'pos=RA,Dec'.")
            return

        data_filename = f'data/dm_vs_baryonic_density_data_RA{asc}_Dec{dec}_Area{area}_Rad{radius}_Range{distance_range[0]}_{distance_range[1]}.csv'

        if os.path.exists(data_filename):
            print(f"Found cached data! Loading directly from {data_filename}...")
            cached_df = pd.read_csv(data_filename)
            densities = cached_df['Baryonic_Density'].values
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

            print(f"Found {len(event_ras)} events. Fetching eRASS baryonic matter sources...")

            baryonic_matter_provider = BaryonicMatterProvider()
            baryons_df = baryonic_matter_provider.get_sources_in_area(asc, dec, area + radius)

            if baryons_df.empty:
                print("No baryonic sources found in this area to calculate density.")
                return

            print(f"Calculating baryonic flux density within {radius}° of each event...")
            
            event_coords = SkyCoord(ra=event_ras*u.degree, dec=event_decs*u.degree)
            baryon_coords = SkyCoord(ra=baryons_df['RA'].values*u.degree, dec=baryons_df['Dec'].values*u.degree)
            
            # Extract baryonic flux values (fill missing/NaN values with 0)
            baryon_fluxes = baryons_df['Total_Flux'].fillna(0).values

            densities = []
            circle_area = np.pi * (radius ** 2)

            for i in range(len(event_coords)):
                seps = event_coords[i].separation(baryon_coords)
                mask = seps.degree <= radius
                
                # Sum the total flux of baryonic sources within the radius
                total_flux = np.sum(baryon_fluxes[mask])
                baryonic_density = total_flux / circle_area
                densities.append(baryonic_density)

            print(f"Saving computed data to {data_filename}...")
            
            os.makedirs('data', exist_ok=True)
            output_df = pd.DataFrame({
                'RA': event_ras,
                'Dec': event_decs,
                'DM_Excess': event_dms,
                'Baryonic_Density': densities
            })
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

        plt.scatter(filtered_densities, filtered_event_dms, alpha=0.7, c='teal', edgecolor='k')
        
        if len(filtered_densities) > 1:
            z = np.polyfit(filtered_densities, filtered_event_dms, 1)
            p = np.poly1d(z)
            r = np.corrcoef(filtered_event_dms, p(filtered_densities))[0, 1]
            r_squared = r**2
            plt.plot(filtered_densities, p(filtered_densities), "r--", alpha=0.8, label=f'Linear Trend (R²={r_squared:.2f})')

        plt.xlabel(f'Local Baryonic Flux Density (Flux / sq degree)\nwithin {radius}° radius')
        plt.ylabel('Dispersion Measure Excess (pc cm$^{-3}$)')
        plt.title(f'DM Excess vs Local Baryonic Flux Density\nCentered at RA={asc}, Dec={dec}')
        plt.legend()
        
        os.makedirs('figures', exist_ok=True)
        plot_filename = f'figures/dm_vs_baryonic_density_plot_RA{asc}_Dec{dec}_Area{area}_Rad{radius}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {plot_filename}")
        plt.show()