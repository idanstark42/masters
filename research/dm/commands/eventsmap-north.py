import numpy as np
import matplotlib.pyplot as plt
from commands.command import Command
from utils import RIGHT_ASCENSION_FIELD, DECLENATION_FIELD

class EventsmapNorthCommand(Command):
    def run(self, args):
        ras, decs, dm_excs = [], [], []
        for ev in self.iterate_events():
            ras.append(float(ev[RIGHT_ASCENSION_FIELD]))
            decs.append(float(ev[DECLENATION_FIELD]))
            dm_excs.append(float(ev["dm_exc"]))

        ras = np.array(ras)
        decs = np.array(decs)
        dm_excs = np.array(dm_excs)

        # Convert RA to radians for the polar plot
        theta = np.radians(ras)
        # Radius is the distance from the North Pole (90 degrees)
        r = 90 - decs

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
        
        # Plot the events
        scatter = ax.scatter(theta, r, s=dm_excs / np.max(dm_excs) * 100, alpha=0.5)

        # Configure the polar axes for standard astronomical conventions
        ax.set_theta_zero_location("N")  # 0h / 0° RA at the top
        ax.set_theta_direction(-1)       # RA increases clockwise when looking at the North pole

        # Set RA tick labels (convert radians back to hours or degrees)
        ax.set_xticks(np.radians(np.arange(0, 360, 45)))
        ax.set_xticklabels([f"{d}°" for d in np.arange(0, 360, 45)])

        # Set Declination tick labels (0 radius = 90 Dec, 90 radius = 0 Dec)
        ax.set_ylim(0, 90)
        ax.set_yticks([0, 30, 60, 90])
        ax.set_yticklabels(['90°', '60°', '30°', '0°'])

        plt.title('FRB Locations (North Celestial Pole Center)', va='bottom')
        plt.savefig('figures/events_map_north.png', dpi=300)
        plt.show()