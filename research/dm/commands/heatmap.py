import numpy as np
import matplotlib.pyplot as plt
from commands.command import Command
from utils import RIGHT_ASCENSION_FIELD, DECLENATION_FIELD

SIGMA = 3.0  

class HeatmapCommand(Command):
    def run(self, args):
        ras, decs, dm_excs = [], [], []
        for ev in self.iterate_events():
            ras.append(float(ev[RIGHT_ASCENSION_FIELD]))
            decs.append(float(ev[DECLENATION_FIELD]))
            dm_excs.append(float(ev["dm_exc"]))

        grid_resolution = 1.0  
        x = np.arange(0, 360, grid_resolution)
        y = np.arange(0, 90, grid_resolution)
        X, Y = np.meshgrid(x, y)
        
        heatmap = np.zeros_like(X, dtype=np.float64)

        for ra, dec, dm in zip(ras, decs, dm_excs):
            # Shortest distance in RA (handles the 0/360 wrap-around)
            dx = (X - ra)
            dx = (dx + 180) % 360 - 180
            
            # Apply polar convergence: RA lines squeeze together at higher declinations
            # We scale the RA distance by cos(Dec) of the grid points
            dx = dx * np.cos(np.radians(Y))
            
            # Declination distance (linear)
            dy = Y - dec
            
            # Add the Gaussian
            heatmap += dm * np.exp(-(dx**2 + dy**2) / (2 * SIGMA**2))

        # 4. Plot the resulting sum
        plt.figure(figsize=(18, 6))
        
        # Standard astronomical plot: RA usually increases right-to-left
        plt.imshow(heatmap, origin='lower', extent=[360, 0, 0, 90], 
                   cmap='magma', aspect='auto')
        
        plt.colorbar(label='Summed Excess DM')
        
        # Invert x-axis to follow standard astronomical convention (RA increases to the East/Left)
        # plt.gca().invert_xaxis()  # Alternative to reversing 'extent' above
        
        plt.xlabel('Right Ascension (deg)')
        plt.ylabel('Declination (deg)')
        plt.title('FRB Excess DM Heatmap')
        plt.savefig('figures/events_heatmap.png', dpi=300)
        plt.show()