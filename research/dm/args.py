import numpy as np

class Arguments:
    def __init__(self, args=None):
        self.args = args if args is not None else []
        self._apply_defaults()
        self._parse()
        self._post_process()

    def _apply_defaults(self):
        self.pos = "180,0"
        self.area = 180.0
        self.radius = 0.5
        self.range = "0,inf"
        self.sigma = 1.0
        self.normalize = False

        self.graining = "grid"
        self.res = 1.0
        self.north = False
        self.south = False

    def _parse(self):
        key_value_pairs = [arg.split('=') for arg in self.args if '=' in arg]
        flags = [arg for arg in self.args if '=' not in arg]

        for key, value in key_value_pairs:
            if hasattr(self, key):
                value = value.strip('"').strip("'")  # Remove quotes if present
                value = type(getattr(self, key))(value)
            setattr(self, key, value)
        
        for flag in flags:
            setattr(self, flag, True)

    def _post_process(self):
        # Convert pos to tuple of floats
        if isinstance(self.pos, str):
            self.pos = tuple(map(float, self.pos.split(',')))
            self.asc, self.dec = self.pos

        # Convert area and radius to float
        self.area = float(self.area) if isinstance(self.area, str) else self.area
        self.radius = float(self.radius) if isinstance(self.radius, str) else self.radius

        # Convert range to tuple of floats
        if isinstance(self.range, str):
            range_values = self.range.split(',')
            self.range = (float(range_values[0]), float(range_values[1]) if range_values[1] != 'inf' else np.inf)

        self.polar = self.north or self.south

    def __getattr__(self, name):
        return False
