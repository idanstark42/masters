import numpy as np
import matplotlib.pyplot as plt
from commands.command import Command
from variables import VARIABLE_TYPES

class CorrelateCommand(Command):
    def run(self, args):
        asc, dec, area, radius = parse_area_args(args)
        distance_range = parse_range(args)
        graining = "gaussian" if "graining=gaussian" in args else "grid"

        selected_vars = [v for v in VARIABLE_TYPES.keys() if v in args]
        
        if len(selected_vars) != 2:
            print("Error: Please provide exactly two variables to correlate.")
            print(f"Available variables: {list(VARIABLE_TYPES.keys())}")
            return
        
        var1, var2 = VARIABLE_TYPES[selected_vars[0]], VARIABLE_TYPES[selected_vars[1]]

        grid_resolution = 1.0 
        ra_min, ra_max = max(asc - area, 0), min(asc + area, 360)
        dec_min, dec_max = max(dec - area, -90), min(dec + area, 90)
        
        ra_edges = np.arange(ra_min, ra_max + grid_resolution, grid_resolution)
        dec_edges = np.arange(dec_min, dec_max + grid_resolution, grid_resolution)

        print(f"Extracting {var1['name']} ({graining})...")
        data1 = var1['extractor']().extract(self, asc, dec, area, distance_range, ra_edges, dec_edges, graining).flatten()
        
        print(f"Extracting {var2['name']} ({graining})...")
        data2 = var2['extractor']().extract(self, asc, dec, area, distance_range, ra_edges, dec_edges, graining).flatten()

        # Clean data: The variables module now outputs NaNs for areas with 0 overlap.
        # This mask safely drops the empty regions.
        valid_mask = np.isfinite(data1) & np.isfinite(data2)
        x = data1[valid_mask]
        y = data2[valid_mask]

        if len(x) < 2:
            print("Not enough valid data points overlapping in the specified area to correlate.")
            return

        m, b = np.polyfit(x, y, 1)
        corr_coef = np.corrcoef(x, y)[0, 1]

        plt.figure(figsize=(10, 8))
        plt.scatter(x, y, alpha=0.5, label=f'Sampled Regions ({graining})')
        
        fit_line = m * x + b
        plt.plot(x, fit_line, color='red', 
                 label=f'Linear Fit: y = {m:.2e}x + {b:.2e}\nPearson R = {corr_coef:.3f}')
        
        plt.xlabel(var1['name'])
        plt.ylabel(var2['name'])
        plt.title(f'Correlation: {var1["name"]} vs {var2["name"]}\nCentered at RA={asc}, Dec={dec}, Area={area}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        filename = f'figures/correlate_{var1["name"]}_{var2["name"]}_RA{asc}_Dec{dec}(graining={graining}).png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
        plt.show()