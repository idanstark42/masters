import csv
import tqdm
from utils import *

class EventsManager:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.repeaters = set()

    def quality_event_base(self, row):
        if row.get(EXCLUDE_FIELD) == '1':
            return False
        if float(row.get(SUB_NUM_FIELD, 0)) > 0:
            return False
        if float(row.get(BONSAI_SNR_FIELD, 0)) < SNR_THRESHOLD:
            return False
        if abs(float(row.get(GALACTIC_LATITUDE_FIELD, 0))) < GALACTIC_LATITUDE_THRESHOLD:
            return False
        
        # Handle repeaters
        repeater_name = row.get(REPEATER_NAME_FIELD)
        if repeater_name:
            if repeater_name in self.repeaters:
                return False
            self.repeaters.add(repeater_name)
            
        return True

    def iterate_events(self):
        with open(self.csv_file, mode='r', encoding='utf-8') as file:
            dict_reader = csv.DictReader(file)
            rows = list(dict_reader)
            
            for row in tqdm.tqdm(rows, desc="Processing base events", ncols=80, unit="event"):
                if self.quality_event_base(row):
                    yield row