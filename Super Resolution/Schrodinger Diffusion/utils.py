class JupyterArgParser:
    def __init__(self):
        self.args = {}

    def add_argument(self, name, type=None, default=None, help=None, action=None):
        self.args[name.lstrip('--').replace('-', '_')] = {
            'type': type,
            'value': default,
            'action': action,
        }

    def set_value(self, name, value):
        if name in self.args:
            self.args[name]['value'] = value
        else:
            raise KeyError(f"Argument '{name}' not found.")
        
    def get_options(self):
        opts = Opt()
        for name, config in self.args.items():
            value = config['value'] if config['value'] is not None else False
            setattr(opts, name, value)

        return opts

class Opt:
    """
    A simple class to hold parsed options.
    """
    def __init__(self):
        pass
