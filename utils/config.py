import yaml
import os


class Config(object):
    """Configuration for the NetFrame applications."""

    def __init__(self, version, config_yaml_path, debug=False):
        super(Config, self).__init__()
        self.version = str(version)
        self.debug = debug
        if '_' in self.version:
            self.sub_version = self.version.split('_')[-1]
        self.set(config_yaml_path)

        # Set default modules to import
        if not hasattr(self, 'data_module'):
            self.data_module = '{0}.{0}_data'.format(self.project)
        if not hasattr(self, 'model_module'):
            self.model_module = '{0}.{0}_model'.format(self.project)

    def set(self, config_yaml_path):
        """Loads parameters from configuartion file"""
        with open(config_yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        if 'experiments' in cfg.keys():
            exp_cfg = cfg['experiments'][int(self.sub_version)]
            # Update the parameters from sub version configuration
            if exp_cfg is not None:
                for key, value in exp_cfg.items():
                    cfg[key] = exp_cfg[key]

        self.__dict__.update(**cfg)

    def update(self, arg_dict):
        """Update parameters"""
        self.__dict__.update(**arg_dict)

    def save(self, config_yaml_path):
        """Saves parameters to configuartion file"""
        with open(config_yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f)

    @property
    def dict(self):
        return self.__dict__

