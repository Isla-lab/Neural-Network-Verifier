from utils.general_utils import *

if __name__ == '__main__':
    
    # read the configuration file to run the verification process
    config = read_config('config.yaml')

    # automatically instanciate the selected verifier
    verifier = instantiate_verifier(config)

    # start the verification process
    #verifier.verify(verbose=1)
    bounds = verifier.compute_bounds()
    print(bounds)
    quit()

    # report the final results 
    verifier.print_results()
