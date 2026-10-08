import numpy as np
import matplotlib.pyplot as plt
from commands.command import Command
from utils import RIGHT_ASCENSION_FIELD, DECLENATION_FIELD

class HeatmapNorthCommand(Command):
    def run(self, args):
        ras, decs, dm_excs = [], [], []
        for ev in self.iterate_events():
            ras.append(float(ev[RIGHT_ASCENSION_FIELD]))
            decs.append(float(ev[DECLENATION_FIELD]))
            dm_excs.append(float(ev["dm_exc"]))

        # 1. Define the 2D grid 
        # Using 361 to ensure the mesh fully closes the circle at 360 degrees
        grid_resolution = 0.5
        x = np.arange(0, 361, grid_resolution)
        y = np.arange(0, 91, grid_resolution)
        X, Y = np.meshgrid(x, y)
        
        dm_heatmap = np.zeros_like(X, dtype=np.float64)
        weight_heatmap = np.zeros_like(X, dtype=np.float64)

        sigma = 2.0  

        # 2. Add Gaussians to both the DM-weighted map and the pure event weight map
        for ra, dec, dm in zip(ras, decs, dm_excs):
            # Shortest distance in RA with polar convergence
            dx = (X - ra)
            dx = (dx + 180) % 360 - 180
            dx = dx * np.cos(np.radians(Y))
            
            # Declination distance
            dy = Y - dec
            
            # Gaussian weight
            gauss = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
            
            dm_heatmap += dm * gauss
            weight_heatmap += gauss

        # 3. Divide to get the DM Density (Average DM per region)
        # We use np.nan for regions with almost no Gaussian overlap so they appear blank
        density_map = np.full_like(X, np.nan)
        valid_regions = weight_heatmap > 1e-4
        invalid_regions = weight_heatmap <= 1e-4
        density_map[valid_regions] = dm_heatmap[valid_regions] / weight_heatmap[valid_regions]
        density_map[invalid_regions] = np.nan  # Explicitly set invalid regions to NaN for clarity

        # 4. Plot on a polar axis
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
        
        # Convert grid for polar plot: angle = RA, radius = distance from North Pole
        theta = np.radians(X)
        r = 90 - Y
        
        # pcolormesh maps the grid coordinates to the density values
        c = ax.pcolormesh(theta, r, density_map, cmap='magma', shading='nearest')
        
        plt.colorbar(c, label='Excess DM')

        # Configure standard astronomical polar orientations
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        # Set RA tick labels
        ax.set_xticks(np.radians(np.arange(0, 360, 45)))
        ax.set_xticklabels([f"{d}°" for d in np.arange(0, 360, 45)])

        # Set Declination tick labels (0 radius = 90 Dec, 90 radius = 0 Dec)
        ax.set_ylim(0, 90)
        ax.set_yticks([0, 30, 60, 90])
        ax.set_yticklabels(['90°', '60°', '30°', '0°'])

        plt.title('FRB DM Density Heatmap', va='bottom', pad=20)
        plt.savefig('figures/dm_density_heatmap_north.png', dpi=300)
        plt.show()