from utils.general_utils import *

if __name__ == '__main__':
    # read the configuration file to run the verification process
    config = read_config('config.yaml')
    properties = create_property(config)

    for prop in properties:
        # automatically instantiate the selected verifier
        verifier = instantiate_verifier(config, prop)

        # compute bounds for the selected verifier
        # bounds = verifier.compute_bounds()
        # print(bounds)

        # start the verification process
        verifier.verify()

        # report the final results
        verifier.print_results()
