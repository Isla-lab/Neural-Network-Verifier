from utils.general_utils import *

if __name__ == '__main__':
    
    # read the configuration file to run the verification process
    config = read_config('config.yaml')

    # automatically instanciate the selected verifier
    verifier = instantiate_verifier(config)

    # compute bounds for the selected verifier
    bounds = verifier.compute_bounds()
    print(bounds)

    # start the verification process
    verifier.verify(verbose=1)

    # report the final results 
    verifier.print_results()
