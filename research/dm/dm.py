import sys
import os
import pandas as pd
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from events_manager import EventsManager
from galaxy_dm_decorator import GalaxyDMDecorator
from galaxies_decorator import GalaxiesProvider
from astropy.coordinates import SkyCoord
import astropy.units as u
import csv

import sys
import os
import inspect
import importlib.util
from commands.command import Command
COMMANDS_DIR = "commands"

def load_commands():
    """Dynamically loads all Command classes in the commands directory."""
    commands = {}
    
    if not os.path.exists(COMMANDS_DIR):
        os.makedirs(COMMANDS_DIR)
        print(f"Created '{COMMANDS_DIR}/' directory. Please add command files.")
        return commands

    for filename in os.listdir(COMMANDS_DIR):
        if filename.endswith(".py") and not filename.startswith("__") and not filename == "command.py":
            command_name = filename[:-3].replace("_", "-")
            file_path = os.path.join(COMMANDS_DIR, filename)
            
            # Dynamically load the module
            spec = importlib.util.spec_from_file_location(command_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Scan the file for any class that inherits from Command
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Command) and obj is not Command:
                    # Instantiate the class and map it to the command name
                    commands[command_name] = obj() 
                    break
            else:
                print(f"Warning: {filename} does not contain a valid Command class.")
                
    return commands

def main(command_name, args):
    commands = load_commands()
    
    if command_name in commands:
        commands[command_name].run(args)
    else:
        print(f"Unknown command: '{command_name}'\n")
        print("Available commands:")
        for cmd in sorted(commands.keys()):
            print(f"  - {cmd}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dm.py <command> [args...]")
    else:
        main(sys.argv[1], sys.argv[2:])