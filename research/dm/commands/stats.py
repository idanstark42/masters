import numpy as np
from commands.command import Command
from utils import parse_area_args, RIGHT_ASCENSION_FIELD, DECLENATION_FIELD

class StatsCommand(Command):
    def run(self, args):
        asc, dec, area, radius = parse_area_args(args)
        dm_excs = []
        dm_excs_all = []

        for ev in self.iterate_events():
            dm_val = float(ev["dm_exc"])
            dm_excs_all.append(dm_val)
            if asc is not None and dec is not None:
                if abs(float(ev[RIGHT_ASCENSION_FIELD]) - asc) < area and abs(float(ev[DECLENATION_FIELD]) - dec) < area:
                    dm_excs.append(dm_val)

        if asc is not None:
            print("--- Events in area ---")
            print(f"Number of events: {len(dm_excs)}")
            print(f"Mean DM excess: {np.mean(dm_excs):.2f}")
            print(f"Median DM excess: {np.median(dm_excs):.2f}")
            print(f"DM excess std: {np.std(dm_excs):.2f}")
        
        print("\n--- All events ---")
        print(f"Number of events: {len(dm_excs_all)}")
        print(f"Mean DM excess: {np.mean(dm_excs_all):.2f}")
        print(f"Median DM excess: {np.median(dm_excs_all):.2f}")
        print(f"DM excess std: {np.std(dm_excs_all):.2f}")