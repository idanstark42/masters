import numpy as np
import matplotlib.pyplot as plt
from commands.command import Command
from utils import parse_area_args, RIGHT_ASCENSION_FIELD, DECLENATION_FIELD

class DmExcHistogramCommand(Command):
    def run(self, args):
        asc, dec, area, radius = parse_area_args(args)
        dm_excs = []

        for ev in self.iterate_events():
            if asc is not None and dec is not None:
                if abs(float(ev[RIGHT_ASCENSION_FIELD]) - asc) > area or abs(float(ev[DECLENATION_FIELD]) - dec) > area:
                    continue
            dm_excs.append(float(ev["dm_exc"]))
        
        bins = np.logspace(np.log10(np.min(dm_excs)), np.log10(np.max(dm_excs)), 61)

        plt.figure(figsize=(12, 6))
        plt.hist(dm_excs, bins=bins, alpha=0.7)
        plt.xscale('log')
        plt.xlabel('Dispersion Measure Excess (pc cm$^{-3}$)')
        plt.ylabel('Number of Events')
        plt.title('Histogram of Dispersion Measure Excess')
        plt.savefig('dm_exc_histogram.png', dpi=300)
        plt.show()