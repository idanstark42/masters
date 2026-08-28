import numpy as np
import matplotlib.pyplot as plt
from commands.command import Command
from utils import RIGHT_ASCENSION_FIELD, DECLENATION_FIELD

class EventsMapCommand(Command):
    def run(self, args):
        ras, decs, dm_excs = [], [], []
        for ev in self.iterate_events():
            ras.append(float(ev[RIGHT_ASCENSION_FIELD]))
            decs.append(float(ev[DECLENATION_FIELD]))
            dm_excs.append(float(ev["dm_exc"]))

        plt.figure(figsize=(12, 6))
        plt.scatter(ras, decs, s=np.array(dm_excs) / np.max(dm_excs) * 100, alpha=0.5)
        plt.xlim(0, 360)
        plt.ylim(0, 90)
        plt.xlabel('Right Ascension (deg)')
        plt.ylabel('Declination (deg)')
        plt.title('Event Locations')
        plt.savefig('events_map.png', dpi=300)
        plt.show()