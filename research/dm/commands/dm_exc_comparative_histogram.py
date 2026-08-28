import matplotlib.pyplot as plt
from commands.command import Command
from utils import parse_area_args, RIGHT_ASCENSION_FIELD, DECLENATION_FIELD

class DmExcComparativeHistogramCommand(Command):
    def run(self, args):
        asc, dec, area, radius = parse_area_args(args)
        if asc is None or dec is None:
            print("No position specified. Use 'pos=RA,Dec'.")
            return

        dm_excs, dm_axcs_in_area = [], []
        for ev in self.iterate_events():
            dm_val = float(ev["dm_exc"])
            dm_excs.append(dm_val)
            if abs(float(ev[RIGHT_ASCENSION_FIELD]) - asc) <= area and abs(float(ev[DECLENATION_FIELD]) - dec) <= area:
                dm_axcs_in_area.append(dm_val)

        plt.figure(figsize=(12, 6))
        plt.ylabel('Density')
        plt.hist(dm_excs, bins=60, alpha=0.7, label='All Events', density=True)
        plt.hist(dm_axcs_in_area, bins=60, alpha=0.7, color='orange', label=f'Events in Area around RA={asc}, Dec={dec}', density=True)
        plt.xlabel('Dispersion Measure Excess (pc cm$^{-3}$)')
        plt.title('Comparative Histogram of Dispersion Measure Excess')
        plt.legend()
        plt.savefig(f'dm_exc_comparative_histogram_RA{asc}_Dec{dec}_Area{area}.png', dpi=300)
        plt.show()