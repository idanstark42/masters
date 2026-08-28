from events_manager import EventsManager
from galaxy_dm_decorator import GalaxyDMDecorator
from utils import CSV_FILE, GALACTIC_DM_MAP_FILE

class Command:
    def iterate_events(self):
        base_manager = EventsManager(CSV_FILE)
        dm_decorator = GalaxyDMDecorator(base_manager, GALACTIC_DM_MAP_FILE)
        return dm_decorator.iterate_events()

    def run(self, args):
        raise NotImplementedError("Commands must implement the run() method.")