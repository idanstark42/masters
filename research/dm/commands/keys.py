import csv
from commands.command import Command
from utils import CSV_FILE

class KeysCommand(Command):
    def run(self, args):
        with open(CSV_FILE, mode='r', encoding='utf-8') as file:
            print(csv.DictReader(file).fieldnames)