import matplotlib.pyplot as plt
from commands.command import Command
from utils import parse_range
from baryonic_matter_decorator import BaryonicMatterProvider

class BaryonsMapCommand(Command):
    def run(self, args):
        baryonic_matter_provider = BaryonicMatterProvider()
        baryons_df = baryonic_matter_provider.get_sources_in_area(asc=180.0, dec=0.0, area=180.0)
        
        if baryons_df.empty:
            print("No baryons were successfully parsed or found within the range!")
            return
            
        baryons_sample = baryons_df.sample(frac=10000/len(baryons_df)) if len(baryons_df) > 10000 else baryons_df
        baryons_sizes = baryons_sample['Total_Flux'] / np.max(baryons_sample['Total_Flux']) * BACKGROUND_DOT_SIZE if 'Total_Flux' in baryons_sample.columns else 20
        print(f"Successfully loaded {len(baryons_sample)} baryons (1% sample). Generating plot...")
        
        plt.figure(figsize=(12, 6))
        plt.scatter(baryons_sample['RA'], baryons_sample['Dec'], s=baryons_sizes, alpha=0.5, c='black')
        plt.xlabel('Right Ascension (deg)')
        plt.ylabel('Declination (deg)')
        plt.title(f'All-Sky Map of eRASS Baryons')
        plt.xlim(0, 360)
        plt.ylim(-90, 90)
        plt.grid(True, alpha=0.3)
        
        filename = f'figures/all_baryons_map.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
        plt.show()