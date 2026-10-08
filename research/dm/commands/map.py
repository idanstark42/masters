import numpy as np
import matplotlib.pyplot as plt
from commands.command import Command
from utils import RIGHT_ASCENSION_FIELD, DECLENATION_FIELD
from args import Arguments
from galaxies_decorator import RegladeGalaxyProvider
from baryonic_matter_decorator import BaryonicMatterProvider

BACKGROUND_DOT_SIZE = 20  # Default size for background points (galaxies and baryons)
EVENT_DOT_SIZE = 100

class MapCommand(Command):
    def run(self, args):
        args = Arguments(args)

        # If polar, we need to fetch the entire RA range (0-360) and filter Dec later.
        # Passing asc=180, dec=0, area=180 will effectively fetch the whole catalog bounding box.
        if args.is_polar:
            fetch_asc, fetch_dec, fetch_area = 180, 0, 180
        else:
            fetch_asc, fetch_dec, fetch_area = args.asc, args.dec, args.area

        fig = plt.figure(figsize=(10, 8))
        
        # Setup projection
        if args.polar:
            ax = fig.add_subplot(111, projection='polar')
            # Optional: Make RA go clockwise, and put 0 degrees at the top
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
        else:
            ax = fig.add_subplot(111)

        # ---------------------------------------------------------
        # EVENTS
        # ---------------------------------------------------------
        ras, decs, dm_excs = [], [], []
        if args.events:
            for ev in self.iterate_events():
                ev_ra = float(ev[RIGHT_ASCENSION_FIELD])
                ev_dec = float(ev[DECLENATION_FIELD])
                
                if args.north and ev_dec < 0:
                    continue
                if args.south and ev_dec > 0:
                    continue
                if not args.polar:
                    if abs(ev_ra - args.asc) > args.area or abs(ev_dec - args.dec) > args.area:
                        continue
                
                ras.append(ev_ra)
                decs.append(ev_dec)
                dm_excs.append(float(ev["dm_exc"]))
                
            if len(ras) > 0:
                print(f"Found {len(ras)} events in the specified area.")
                
                plot_ras, plot_decs = ras, decs
                if args.polar:
                    plot_ras = np.radians(ras)
                    # For polar: r is the distance from the pole. 
                    # North center=90 (so r=90-dec). South center=-90 (so r=dec+90).
                    plot_decs = [90 - d if args.north else d + 90 for d in decs]
                
                ax.scatter(
                    plot_ras, plot_decs, 
                    s=np.array(dm_excs) / np.max(dm_excs) * EVENT_DOT_SIZE if len(dm_excs) > 0 else 20, 
                    alpha=0.8,
                    zorder=3, # Bring events to the front
                    label='FRBs (Size = DM)'
                )
            else:
                print("No events found in the specified area.")

        # ---------------------------------------------------------
        # GALAXIES
        # ---------------------------------------------------------
        if args.galaxies:
            galaxies_provider = RegladeGalaxyProvider()
            galaxies_df = galaxies_provider.get_galaxies_in_area(fetch_asc, fetch_dec, fetch_area, distance_range)
            
            if args.north:
                galaxies_df = galaxies_df[galaxies_df['Dec'] >= 0]
            elif args.south:
                galaxies_df = galaxies_df[galaxies_df['Dec'] <= 0]
                
            galaxies_sample = galaxies_df.sample(frac=10000/len(galaxies_df)) if len(galaxies_df) > 10000 else galaxies_df
            
            if not galaxies_sample.empty:
                print(f"using mass: {'yes' if 'M' in galaxies_sample.columns else 'no'}")
                if 'M' in galaxies_sample.columns:
                    print(f"min: {galaxies_sample['M'].min()}, max: {galaxies_sample['M'].max()}")
                
                g_ras = galaxies_sample['RA'].values
                g_decs = galaxies_sample['Dec'].values
                if args.polar:
                    g_ras = np.radians(g_ras)
                    g_decs = 90 - g_decs if args.north else g_decs + 90
                
                g_sizes = galaxies_sample['M'] / np.max(galaxies_sample['M']) * BACKGROUND_DOT_SIZE if 'M' in galaxies_sample.columns else 20
                ax.scatter(g_ras, g_decs, s=g_sizes, alpha=0.5, c='black', label='Galaxies', zorder=1)
                print(f"Successfully loaded {len(galaxies_sample)} galaxies.")
            else:
                print("No galaxies found in the specified area.")

        # ---------------------------------------------------------
        # BARYONIC MATTER
        # ---------------------------------------------------------
        if args.baryons:
            baryonic_matter_provider = BaryonicMatterProvider()
            baryons_df = baryonic_matter_provider.get_sources_in_area(asc=fetch_asc, dec=fetch_dec, area=fetch_area)
            
            if args.north:
                baryons_df = baryons_df[baryons_df['Dec'] >= 0]
            elif args.south:
                baryons_df = baryons_df[baryons_df['Dec'] <= 0]

            baryon_sample = baryons_df.sample(frac=10000/len(baryons_df)) if len(baryons_df) > 10000 else baryons_df
            
            if not baryon_sample.empty:
                b_ras = baryon_sample['RA'].values
                b_decs = baryon_sample['Dec'].values
                if args.polar:
                    b_ras = np.radians(b_ras)
                    b_decs = 90 - b_decs if args.north else b_decs + 90
                    
                b_sizes = baryon_sample['Total_Flux'] / np.max(baryon_sample['Total_Flux']) * BACKGROUND_DOT_SIZE if 'Total_Flux' in baryon_sample.columns else 20
                ax.scatter(b_ras, b_decs, s=b_sizes, alpha=0.5, c='red', label='Baryonic Sources', zorder=2)
                print(f"Successfully loaded {len(baryon_sample)} baryonic sources.")
            else:
                print("No baryonic sources found in the specified area.")

        # ---------------------------------------------------------
        # AXIS FORMATTING
        # ---------------------------------------------------------
        if args.polar:
            # Set the radial limits from the pole (0) to the equator (90 degrees away)
            ax.set_ylim(0, 90)
            
            # Format radial (Declination) ticks
            ax.set_yticks([0, 30, 60, 90])
            if args.north:
                ax.set_yticklabels(['90°', '60°', '30°', '0°'])
                plt.title(f'FRBs, Galaxies & Baryonic Matter\nNorthern Hemisphere')
            else:
                ax.set_yticklabels(['-90°', '-60°', '-30°', '0°'])
                plt.title(f'FRBs, Galaxies & Baryonic Matter\nSouthern Hemisphere')
                
            # Format angular (Right Ascension) ticks
            ax.set_xticks(np.radians(np.arange(0, 360, 45)))
            ax.set_xticklabels([f'{angle}°' for angle in np.arange(0, 360, 45)])
            
            filename = f'figures/area_map_{"North" if args.north else "South"}Hemisphere_Range{args.range[0]}_{args.range[1]}.png'
        else:
            ax.set_xlim(np.max([asc - area, 0]), np.min([asc + area, 360]))
            ax.set_ylim(np.max([dec - area, -90]), np.min([dec + area, 90]))
            ax.set_xlabel('Right Ascension (deg)')
            ax.set_ylabel('Declination (deg)')
            ax.set_title(f'FRBs, Galaxies & Baryonic Matter\nCentered at RA={asc}, Dec={dec}')
            
            filename = f'figures/area_map_RA{asc}_Dec{dec}_Area{area}_Range{args.range[0]}_{args.range[1]}.png'

        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
        ax.grid(True, alpha=0.3)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
        plt.show()