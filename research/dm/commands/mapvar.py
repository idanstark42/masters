import numpy as np
import matplotlib.pyplot as plt
from commands.command import Command
from utils import parse_area_args, parse_range
from variables import VARIABLE_TYPES

class MapVarCommand(Command):
    def run(self, args):
        asc, dec, area, radius = parse_area_args(args)
        distance_range = parse_range(args)
        graining = "gaussian" if "graining=gaussian" in args else "grid"

        selected_vars = [v for v in VARIABLE_TYPES.keys() if v in args]
        
        if not selected_vars:
            print("Error: Please provide at least one variable to map.")
            print(f"Available variables: {list(VARIABLE_TYPES.keys())}")
            return

        grid_resolution = 1.0
        ra_min, ra_max = max(asc - area, 0), min(asc + area, 360)
        dec_min, dec_max = max(dec - area, -90), min(dec + area, 90)
        
        ra_edges = np.arange(ra_min, ra_max + grid_resolution, grid_resolution)
        dec_edges = np.arange(dec_min, dec_max + grid_resolution, grid_resolution)

        # Layout: one row, multiple columns based on how many variables were requested
        fig, axes = plt.subplots(1, len(selected_vars), figsize=(7 * len(selected_vars), 6), squeeze=False)
        axes = axes.flatten()

        for ax, var_key in zip(axes, selected_vars):
            var = VARIABLE_TYPES[var_key]
            print(f"Extracting {var['name']} ({graining})...")
            
            data = var['extractor']().extract(self, asc, dec, area, distance_range, ra_edges, dec_edges, graining)
            
            # data is shape (len(RA), len(Dec)). Transpose (.T) to match imshow's (Y, X) requirement.
            im = ax.imshow(data.T, origin='lower', extent=[ra_min, ra_max, dec_min, dec_max], 
                           cmap='magma', aspect='auto')
            
            fig.colorbar(im, ax=ax, label=var['name'])
            ax.set_xlabel('Right Ascension (deg)')
            ax.set_ylabel('Declination (deg)')
            ax.set_title(f"{var['name']} ({graining.title()})")
            
        plt.suptitle(f'Mapped Variables Centered at RA={asc}, Dec={dec}, Area={area}°', y=1.02)
        plt.tight_layout()
        
        var_names = "_".join(selected_vars)
        filename = f'figures/mapvar_{var_names}_RA{asc}_Dec{dec}(graining={graining}).png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
        plt.show()