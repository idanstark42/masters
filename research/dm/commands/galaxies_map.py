import matplotlib.pyplot as plt
from commands.command import Command
from utils import parse_range
from galaxies_decorator import GalaxiesProvider

class GalaxiesMapCommand(Command):
    def run(self, args):
        distance_range = parse_range(args)
        print(f"Scanning the GLADE+ catalog to map galaxies within distance {distance_range} Mpc...")
        
        galaxies_provider = GalaxiesProvider()
        galaxies_df = galaxies_provider.get_galaxies_in_area(asc=180.0, dec=0.0, area=180.0, distance_range=distance_range)
        
        if galaxies_df.empty:
            print("No galaxies were successfully parsed or found within the range!")
            return
            
        sample = galaxies_df.sample(frac=0.5) if len(galaxies_df) > 10000 else galaxies_df
        sizes = sample['M']
        print(f"Successfully loaded {len(sample)} galaxies (50% sample). Generating plot...")
        
        plt.figure(figsize=(12, 6))
        plt.scatter(sample['RA'], sample['Dec'], s=sizes, alpha=0.5, c='black')
        plt.xlabel('Right Ascension (deg)')
        plt.ylabel('Declination (deg)')
        plt.title(f'All-Sky Map of GLADE+ Galaxies (Distance: {distance_range} Mpc, 1% Subsample)')
        plt.xlim(0, 360)
        plt.ylim(-90, 90)
        plt.grid(True, alpha=0.3)
        
        filename = f'all_galaxies_map_dist_{distance_range[0]}_{distance_range[1]}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
        plt.show()