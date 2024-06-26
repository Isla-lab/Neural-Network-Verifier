import importlib

import yaml

from utils.converters import convert_model, NetVerModel
from utils.yaml_property_reader import config_to_bounds


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def read_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
           
        except yaml.YAMLError as exc:
            print(exc)

    # check the params provided for the algorithm to evaluate
    check_parameters(config)

    # loading the original DNN and convert it into a NetVer format one
    new_path = convert_model(config)
    config['model']['path'] = new_path

    return config


def check_parameters(config_file):
    #verifier to be checked: ["ProVe", "CountingProVe", "eProVe"]
    pass


def create_property(config_file_prop):
    print(f'\rReading safety property...{" " * 20}', end='')
    bounds, compute_output_bounds = config_to_bounds(config_file_prop)
    print(f'\rSafety property loaded!{" " * 20}')
    return bounds, compute_output_bounds


def instantiate_verifier(params, prop):
    verifier = params['verifier']['name']

    try:
        module = importlib.import_module(f"backends.{verifier}")
        method_class = getattr(module, verifier)
        return method_class(params, prop)
    except ModuleNotFoundError:
        raise ImportError(f"Module '{verifier}' not found.")
    except AttributeError:
        raise AttributeError(f"Class '{verifier}' not found in module '{verifier}'.")


def get_netver_model(model, output_bounds):
    if len(output_bounds) > 0:
        return NetVerModel(model, output_bounds)
    else:
        return None
