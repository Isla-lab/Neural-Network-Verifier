
from utils.general_utils import *

from example_models.resnet_model import Net, ResidualBlock

import __main__
setattr(__main__, "Net", Net)
setattr(__main__, "ResidualBlock", ResidualBlock)

if __name__ == '__main__':
    # read the configuration file to run the verification process
    config = read_config('config.yaml')
    properties, compute_bounds = create_property(config)

    verified_properties = []
    for prop_idx, prop in enumerate(properties):
        print(f'\rVerifying property {prop_idx}/{len(properties)}{" " * 20}', end='')
        # automatically instantiate the selected verifier
        verifier = instantiate_verifier(config, prop)

        # compute output bounds for the selected verifier
        if compute_bounds:
            bounds = verifier.compute_bounds()
            print(bounds)
        # verify output SAT condition
        else:
            verified_properties.append(verifier.verify())
            # report the final results
            # verifier.print_results()
    print("\nVerification complete!")

    print_verification_metrics(verified_properties)
