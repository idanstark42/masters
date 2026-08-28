import numpy as np
import matplotlib.pyplot as plt
from commands.command import Command
from utils import DEFAULT_RA_RES, DEFAULT_DEC_RES, MIN_COUNT, RIGHT_ASCENSION_FIELD, DECLENATION_FIELD, DISPERSION_MEASURE_FIELD, DISPERSION_MEASURE_BACKUP_FIELD

class HeatmapCommand(Command):
    def run(self, args):
        ra_res = DEFAULT_RA_RES
        dec_res = DEFAULT_DEC_RES
        min_cnt = MIN_COUNT

        if len(args) > 0:
            ra_res, dec_res, min_cnt = map(int, args[0].split(':'))

        values, ras, decs = [], [], []

        for ev in self.iterate_events():
            ras.append(float(ev[RIGHT_ASCENSION_FIELD]))
            decs.append(float(ev[DECLENATION_FIELD]))
            values.append(float(ev[DISPERSION_MEASURE_FIELD] or ev[DISPERSION_MEASURE_BACKUP_FIELD]))

        ra_edges = np.linspace(0, 360, ra_res + 1)
        dec_edges = np.linspace(0, 90, dec_res + 1)

        sums, _, _ = np.histogram2d(decs, ras, bins=[dec_edges, ra_edges], weights=values)
        counts, _, _ = np.histogram2d(decs, ras, bins=[dec_edges, ra_edges])
        averages = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > min_cnt)

        plt.figure(figsize=(12, 6))
        plt.imshow(averages, origin='lower', aspect='auto', extent=[0, 360, 0, 90])
        plt.colorbar(label='Dispersion Measure')
        plt.xlabel('Right Ascension (deg)')
        plt.ylabel('Declination (deg)')
        plt.title('Average Dispersion Measure Heatmap')
        plt.savefig(f'heatmap_RAres{ra_res}_Decres{dec_res}_Mincount{min_cnt}.png', dpi=300)
        plt.show()