from utils.general_utils import *

from example_models.resnet_model import Net, ResidualBlock

import __main__
setattr(__main__, "Net", Net)
setattr(__main__, "ResidualBlock", ResidualBlock)

if __name__ == '__main__':
    # read the configuration file to run the verification process
    config = read_config('config.yaml')
    properties, compute_bounds = create_property(config)

    for prop in properties:
        # automatically instantiate the selected verifier
        verifier = instantiate_verifier(config, prop)

        # compute output bounds for the selected verifier
        if compute_bounds:
            bounds = verifier.compute_bounds()
            print(bounds)
        # verify output SAT condition
        else:
            verifier.verify()
            # report the final results
            # verifier.print_results()
