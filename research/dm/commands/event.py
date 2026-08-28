import csv
from commands.command import Command
from utils import CSV_FILE

class EventCommand(Command):
    def run(self, args):
        if not args:
            print("Please provide an event index. Example: event 5")
            return
        index = int(args[0])
        with open(CSV_FILE, mode='r', encoding='utf-8') as file:
            rows = list(csv.DictReader(file))
            print(rows[index])