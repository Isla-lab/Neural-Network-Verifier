import yaml
import importlib
import numpy as np
from utils.converters import get_netver_model

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
    new_path = get_netver_model(config)
    config['model']['path'] = new_path

    return config

def check_parameters(config_file):
    #verifier to be checked: ["ProVe", "CountingProVe", "eProVe"]
    pass

def create_property(config_file_prop):
    return np.array(config_file_prop)


def instantiate_verifier(params):
    verifier = params['verifier']['name']

    try:
        module = importlib.import_module(f"backends.{verifier}")
        method_class = getattr(module, verifier)
        return method_class(params)
    except ModuleNotFoundError:
        raise ImportError(f"Module '{verifier}' not found.")
    except AttributeError:
        raise AttributeError(f"Class '{verifier}' not found in module '{verifier}'.")
