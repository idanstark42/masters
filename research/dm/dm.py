import sys
import os
import inspect
import importlib.util
import itertools
from commands.command import Command

COMMANDS_DIR = "commands"
ARG_SEPARATOR = "|"

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

def parse_argument_combinations(args):
    """
    Parses CLI arguments and splits pipe-delimited strings into lists of possible values, 
    preserving the 'key=' prefix if it exists. 
    Returns a list of all possible argument combinations.
    """
    parsed_args = []
    
    for arg in args:
        if '=' in arg:
            key, val_str = arg.split('=', 1)
            # Split the values and re-attach the key to each
            vals = val_str.split(ARG_SEPARATOR)
            parsed_args.append([f"{key}={v}" for v in vals])
        else:
            parsed_args.append(arg.split(ARG_SEPARATOR))
            
    # itertools.product creates the Cartesian product of all argument lists
    # Example: [[A, B], [C, D]] -> [(A,C), (A,D), (B,C), (B,D)]
    return list(itertools.product(*parsed_args))

def main(command_name, args):
    commands = load_commands()
    
    if command_name not in commands:
        print(f"Unknown command: '{command_name}'\n")
        print("Available commands:")
        for cmd in sorted(commands.keys()):
            print(f"  - {cmd}")
        return

    # Generate all combinations of the arguments
    combinations = parse_argument_combinations(args)
    
    print(f"Prepared {len(combinations)} execution(s) for '{command_name}'.")
    
    # Run the command for each combination
    for combo in combinations:
        combo_list = list(combo)
        if len(combinations) > 1:
            print(f"\n[{command_name}] Running with arguments: {combo_list}")
            print("-" * 50)
            
        commands[command_name].run(combo_list)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dm.py <command> [args...]")
        print(f"Tip: You can use '{ARG_SEPARATOR}' to run multiple combinations, e.g., area=1{ARG_SEPARATOR}5{ARG_SEPARATOR}10")
    else:
        main(sys.argv[1], sys.argv[2:])