import functools
import operator
import os

from utils.vnnlib_parser import parse_vnnlib_files
import torch


class YamlConfigReaderException(Exception):
    """
    Exception raised for errors in the format of YAML config files

    Attributes:
        message -- explanation of the format error
    """

    def __init__(self, message="Badly formatted YAML config file"):
        self.message = message
        super().__init__(self.message)


def convert_string_bounds(yaml_bounds):
    # Convert numerical values for bounds to float type
    new_bounds = []
    for variable_bounds in yaml_bounds:
        float_bounds = []
        for bound in variable_bounds:
            try:
                float_bound = float(bound)
            except TypeError:
                float_bound = bound
            float_bounds.append(float_bound)

        new_bounds.append(float_bounds)
    return [new_bounds]


def pools_to_disjunction(pools, pool_idx, num_nodes, condition):
    # Derive bounds for the specified pools of output nodes
    disjunction_bounds = []

    if not condition.startswith("non-"):
        for node in pools[pool_idx]:
            try:
                node_idx = int(node)
            except TypeError:
                raise YamlConfigReaderException(f"Invalid node index {node} in pool {pool_idx}")

            disjunction_bound = [[float('-inf'), float('inf')] for _ in range(num_nodes)]
            if condition == "positive":
                disjunction_bound[node_idx] = [0.0, float('inf')]
            elif condition == "negative":
                disjunction_bound[node_idx] = [float('-inf'), 0.0]
            elif condition == "max":
                for bound_idx in range(num_nodes):
                    if bound_idx not in pools[pool_idx]:
                        disjunction_bound[bound_idx] = [float('-inf'), f'Y_{node_idx}']
            elif condition == "min":
                for bound_idx in range(num_nodes):
                    if bound_idx not in pools[pool_idx]:
                        disjunction_bound[bound_idx] = [f'Y_{node_idx}', float('inf')]
            elif isinstance(condition, list):
                disjunction_bound[node_idx] = convert_string_bounds([condition])[0][0]
            else:
                raise YamlConfigReaderException(f"Invalid SAT condition: {condition}")
            disjunction_bounds.append(disjunction_bound)
    else:
        for node in range(num_nodes):
            if node not in pools[pool_idx]:
                disjunction_bound = [[float('-inf'), float('inf')] for _ in range(num_nodes)]

                if condition == "non-max":
                    for bound_idx in pools[pool_idx]:
                        disjunction_bound[bound_idx] = [float('-inf'), f'Y_{node}']
                elif condition == "non-min":
                    for bound_idx in pools[pool_idx]:
                        disjunction_bound[bound_idx] = [f'Y_{node}', float('inf')]
                else:
                    raise YamlConfigReaderException(f"Invalid SAT condition: {condition}")

                disjunction_bounds.append(disjunction_bound)

    return disjunction_bounds


def condition_type_to_pools(sat_condition, use_abstraction, num_nodes):
    # Handle concrete and abstract properties to derive output pools
    if not use_abstraction:
        num_pools = num_nodes
        pools = [[node_idx] for node_idx in range(num_pools)]
        pool_idx = int(sat_condition["node_index"])
        condition = sat_condition["node_condition"]
    else:
        pools = sat_condition["pools"]
        pool_idx = int(sat_condition["pool_index"])
        condition = sat_condition["pool_condition"]

    return pools, pool_idx, condition


def prop_to_output_bounds(sat_condition, use_abstraction, num_nodes):
    pools, pool_idx, condition = condition_type_to_pools(sat_condition, use_abstraction, num_nodes)

    output_bounds = pools_to_disjunction(pools, pool_idx, num_nodes, condition)

    return output_bounds


def get_standard_bounds(center):
    bounds = []
    for x in center:
        bounds.append([x, x])
    return bounds


def perturb_bounds(bounds, perturbation, perturbation_type):
    if perturbation_type == "radius":
        bounds[0] -= perturbation
        bounds[1] += perturbation
    elif perturbation_type == "upper_threshold":
        bounds[1] = perturbation
    elif perturbation_type == "lower_threshold":
        bounds[0] = perturbation
    else:
        raise YamlConfigReaderException(f"Invalid perturbation type: {perturbation_type}")


def recurse_over_patch_dimensionality(input_shape, patch_size, current_dim, current_input_bounds, start_indices,
                                      patch_indices, perturbation, perturbation_type):
    if current_dim < len(patch_size):
        for x in range(start_indices[current_dim], start_indices[current_dim] + patch_size[current_dim]):
            current_patch_indices = patch_indices.copy()
            current_patch_indices.append(x)
            recurse_over_patch_dimensionality(input_shape, patch_size, current_dim + 1, current_input_bounds,
                                              start_indices,
                                              current_patch_indices, perturbation, perturbation_type)
    else:
        final_index = 0
        for dim, index in enumerate(patch_indices):
            volume = index
            for other_dim in range(dim + 1, len(patch_indices)):
                volume *= input_shape[other_dim]
            final_index += volume

        perturb_bounds(current_input_bounds[final_index], perturbation, perturbation_type)


def recurse_over_input_dimensionality(input_shape, patch_size, current_dim, input_bounds, start_indices, center,
                                      perturbation, perturbation_type):
    if current_dim < len(input_shape):
        for x in range(input_shape[current_dim] - patch_size[current_dim] + 1):
            current_start_indices = start_indices.copy()
            current_start_indices.append(x)
            recurse_over_input_dimensionality(input_shape, patch_size, current_dim + 1, input_bounds,
                                              current_start_indices, center, perturbation, perturbation_type)
    else:
        current_input_bounds = get_standard_bounds(center)
        recurse_over_patch_dimensionality(input_shape, patch_size, 0, current_input_bounds, start_indices, [],
                                          perturbation, perturbation_type)
        input_bounds.append(current_input_bounds)


def perturbation_to_bounds(domain, input_shape):
    if "perturbation_center" in domain:
        center = domain["perturbation_center"]
    else:
        center_path = domain["perturbation_center_path"]
        center_tensor = torch.load(center_path)
        center = center_tensor.flatten().tolist()

    perturbation = float(domain["perturbation"])
    perturbation_type = domain["perturbation_type"]

    if "patch_size" not in domain:
        input_bounds = []
        for value in center:
            converted_value = float(value)
            bounds = [converted_value, converted_value]
            perturb_bounds(bounds, perturbation, perturbation_type)
            input_bounds.append(bounds)
        input_bounds = [input_bounds]
    else:
        patch_size = domain["patch_size"]

        if len(patch_size) != len(input_shape):
            raise YamlConfigReaderException(f"Invalid patch size {patch_size} for the given input shape {input_shape}")

        input_bounds = []
        recurse_over_input_dimensionality(input_shape, patch_size, 0, input_bounds, [], center, perturbation,
                                          perturbation_type)

    return input_bounds


def prop_to_input_bounds(domain, condition_type, input_shape):
    if condition_type == "safety":
        input_bounds = convert_string_bounds(domain["bounds"])
    elif condition_type == "robustness":
        input_bounds = perturbation_to_bounds(domain, input_shape)
    else:
        raise YamlConfigReaderException(f"Invalid condition type: {condition_type}")
    return input_bounds


def format_file_content(input_bounds, output_bounds, num_output_bounds):
    content = ""

    # Add variable declarations
    for variable, _ in enumerate(input_bounds[0]):
        content += f"(declare-const X_{variable} Real)\n"

    for variable in range(num_output_bounds):
        content += f"(declare-const Y_{variable} Real)\n"

    for input_variable, bound in enumerate(input_bounds[0]):
        content += f"(assert (>= X_{input_variable} {bound[0]}))\n"
        content += f"(assert (<= X_{input_variable} {bound[1]}))\n"

    content += "\n(assert (or\n"
    for output_bound in output_bounds:
        content += "    (and "

        for output_variable, bound in enumerate(output_bound):
            if isinstance(bound[0], str) or bound[0] > float('-inf'):
                content += f"(>= Y_{output_variable} {bound[0]})"
            if isinstance(bound[1], str) or bound[1] < float('inf'):
                content += f"(<= Y_{output_variable} {bound[1]})"

        content += ")\n"
    content += "))\n"

    return content


def format_folder_content(input_bounds, output_bounds, num_output_nodes):
    folder_content = []

    for prop_index, input_bound in enumerate(input_bounds):
        print(f'\rFormatting property {prop_index}/{len(input_bounds)}{" " * 20}', end='')

        content = ""

        # Add variable declarations
        for variable, _ in enumerate(input_bounds[0]):
            content += f"(declare-const X_{variable} Real)\n"

        for variable in range(num_output_nodes):
            content += f"(declare-const Y_{variable} Real)\n"

        for input_variable, bound in enumerate(input_bound):
            content += f"(assert (>= X_{input_variable} {bound[0]}))\n"
            content += f"(assert (<= X_{input_variable} {bound[1]}))\n"

        content += "\n(assert (or\n"
        for output_bound in output_bounds:
            content += "    (and "

            for output_variable, bound in enumerate(output_bound):
                if isinstance(bound[0], str) or bound[0] > float('-inf'):
                    content += f"(>= Y_{output_variable} {bound[0]}) "
                if isinstance(bound[1], str) or bound[1] < float('inf'):
                    content += f"(<= Y_{output_variable} {bound[1]})"

            content += ")\n"
        content += "))\n"

        folder_content.append(content)

    return folder_content


def clear_vnnlib_properties_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            try:
                os.remove(file_path)
            except OSError as e:
                print(f'Failed to delete old VNNLIB file {file_path}: {e}')
    else:
        try:
            os.makedirs(folder_path)
        except OSError as e:
            print(f'Failed to create a new VNNLIB directory {folder_path}: {e}')


def write_to_file(folder_path, content):
    clear_vnnlib_properties_folder(folder_path)

    try:
        file = open(f"./{folder_path}/{folder_path}.vnnlib", 'w')
        file.write(content)
    except Exception as e:
        raise YamlConfigReaderException(
            f"An error occurred while trying to write the VNNLIB file {folder_path}/{folder_path}.vnnlib: {e}")


def write_to_folder(folder_path, folder_content):
    clear_vnnlib_properties_folder(folder_path)

    for content_idx, content in enumerate(folder_content):
        print(f'\rWriting property {content_idx}/{len(folder_content)} to file{" " * 20}', end='')
        try:
            file = open(f"./{folder_path}/{folder_path}_{content_idx}.vnnlib", 'w')
            file.write(content)
        except Exception as e:
            raise YamlConfigReaderException(
                f"An error occurred while trying to write the vnnlib file "
                f"{folder_path}/{folder_path}_{content_idx}.vnnlib: {e}")


def write_vnnlib_file(folder_path, prop, input_shape, output_shape, compute_bounds):
    print(f'\rConverting config property to VNNLIB format...{" " * 20}', end='')

    # Derive input and output bounds
    domain = prop["domain"]
    condition_type = prop["type"]
    input_bounds = prop_to_input_bounds(domain, condition_type, input_shape)

    use_patches = "patch_size" in domain

    num_output_nodes = functools.reduce(operator.mul, output_shape, 1)

    if not compute_bounds:
        sat_condition = prop["sat_condition"]

        use_abstraction = prop["use_abstraction"]

        output_bounds = prop_to_output_bounds(sat_condition, use_abstraction, num_output_nodes)
    else:
        output_bounds = []

    # Format bounds into the VNN-LIB format
    if use_patches:
        vnnlib_folder_content = format_folder_content(input_bounds, output_bounds, num_output_nodes)

        # Write VNN-LIB properties to file
        write_to_folder(folder_path, vnnlib_folder_content)
    else:
        vnnlib_content = format_file_content(input_bounds, output_bounds, num_output_nodes)

        # Write VNN-LIB properties to file
        write_to_file(folder_path, vnnlib_content)


def config_to_bounds(config):
    """
    Method that reads the provided config dictionary and derives input and output bounds for the sat-condition

    Parameters
    ----------
     config : dict
    	 The NetVer config dict

    Compute
    ----------
        properties_bounds : list[dict]
        	A list of dictionaries, each for a distinct property
        	Each dictionary contains the entries "inputs" and "outputs" for the input and output boundaries

    Raises
    ------
        YamlConfigReaderException
            If an error is encountered while reading the config dictionary or for invalid configurations
    """
    prop = config["property"]
    input_shape = config["model"]["input_shape"]
    output_shape = config["model"]["output_shape"]

    compute_output_bounds = "sat_condition" not in prop

    # Extract bounds from a specified VNN-LIB file or write one based on the config file
    if "path" in prop:
        properties_set = parse_vnnlib_files(prop["path"])
    else:
        write_vnnlib_file("./config_properties", prop, input_shape, output_shape, compute_output_bounds)
        properties_set = parse_vnnlib_files("./config_properties")

    # Change dictionary structure into list of lists
    properties_bounds = []
    for properties_idx, properties in enumerate(properties_set):
        print(f'\rDeriving bounds for property {properties_idx}/{len(properties_set)}{" " * 20}', end='')

        for prop in properties:
            property_bounds = {"inputs": [], "outputs": []}

            for variable in prop["inputs"].keys():
                property_bounds["inputs"].append(prop["inputs"][variable])

            for output_disjunction in prop["outputs"]:
                new_output_bounds = []
                for variable in output_disjunction.keys():
                    new_output_bounds.append(output_disjunction[variable])
                property_bounds["outputs"].append(new_output_bounds)

            properties_bounds.append(property_bounds)

    return properties_bounds, compute_output_bounds
