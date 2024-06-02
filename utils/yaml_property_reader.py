from utils.vnnlib_parser import parse_vnnlib_file

class YamlConfigReaderException(Exception):
    """
    Exception raised for errors in the format of YAML config files

    Attributes:
        message -- explanation of the format error
    """
    def __init__(self, message="Badly formatted YAML config file"):
        self.message = message
        super().__init__(self.message)

def fix_yaml_bounds(yaml_bounds):
    # Convert numerical values for bounds to float type
    new_bounds = []
    for variable_bounds in yaml_bounds:
        float_bounds = []
        for bound in variable_bounds:
            try:
                float_bound = float(bound)
            except TypeError:
                float_bound  = bound
            float_bounds.append(float_bound)

        new_bounds.append(float_bounds)
    return new_bounds

def pools_to_disjunction(pools, pool_idx, num_nodes, bounds_type):
    # Derive bounds for the specified pools of output nodes
    disjunction_bounds = []

    for node in pools[pool_idx]:
        try:
            node_idx = int(node)
        except TypeError:
            raise YamlConfigReaderException(f"Invalid node index {node} in pool {pool_idx}")

        disjunction_bound = [[float('-inf'), float('inf')] for _ in range(num_nodes)]
        if bounds_type == "positive":
            disjunction_bound[node_idx] = [0.0, float('inf')]
        elif bounds_type == "negative":
            disjunction_bound[node_idx] = [float('-inf'), 0.0]
        elif bounds_type == "max":
            for bound_idx in range(num_nodes):
                if bound_idx != node_idx:
                    disjunction_bound[bound_idx] = [float('-inf'), f'Y_{node_idx}']
        elif bounds_type == "min":
            for bound_idx in range(num_nodes):
                if bound_idx not in pools[pool_idx]:
                    disjunction_bound[bound_idx] = [f'Y_{node_idx}', float('inf')]
        elif isinstance(bounds_type, list):
            disjunction_bound[node_idx] = fix_yaml_bounds([bounds_type])[0]
        else:
            raise YamlConfigReaderException(f"Invalid bounds for SAT condition: {bounds_type}")
        disjunction_bounds.append(disjunction_bound)

    return disjunction_bounds

def condition_type_to_pools(condition, condition_type):
    # Handle concrete and abstract properties to derive output pools
    if condition_type == "concrete":
        num_nodes = int(condition["num_nodes"])
        num_pools = num_nodes
        pools = [[node_idx] for node_idx in range(num_pools)]
        pool_idx = int(condition["node"])
    elif condition_type == "abstract":
        pools = condition["pools"]
        num_nodes = condition["num_nodes"]
        pool_idx = int(condition["pool"])
    else:
        raise YamlConfigReaderException(f"Invalid type for SAT condition: {condition_type}")

    return pools, pool_idx, num_nodes

def prop_to_output_bounds(condition, condition_type):
    pools, pool_idx, num_nodes = condition_type_to_pools(condition, condition_type)

    bounds_type = condition["bounds"]
    output_bounds = pools_to_disjunction(pools, pool_idx, num_nodes, bounds_type)

    return output_bounds

def prop_to_input_bounds(domain):
    input_bounds = fix_yaml_bounds(domain)
    return input_bounds

def format_file_content(input_bounds, output_bounds):
    content = ""

    # Add variable declarations
    for variable, _ in enumerate(input_bounds):
        content += f"(declare-const X_{variable} Real)\n"

    for variable, _ in enumerate(output_bounds[0]):
        content += f"(declare-const Y_{variable} Real)\n"

    # Add input assertions
    for variable, input_bound in enumerate(input_bounds):
        content += f"(assert(>= X_{variable} {input_bound[0]}))\n"
        content += f"(assert(<= X_{variable} {input_bound[1]}))\n"

    # Add disjunction output assertions
    content += f"(assert (or\n"
    for disjunction_bound in output_bounds:
        content += "    (and "
        for variable, output_bound in enumerate(disjunction_bound):
            if isinstance(output_bound[0], str) or output_bound[0] > float('-inf'):
                content += f"(>= Y_{variable} {output_bound[0]}) "
            if isinstance(output_bound[1], str) or output_bound[1] < float('inf'):
                content += f"(<= Y_{variable} {output_bound[1]})"
        content += ")\n"
    content += "))\n"

    return content

def write_to_file(file_path, content):
    try:
        file = open(file_path, 'w')
        file.write(content)
    except Exception as e:
        raise YamlConfigReaderException(f"An error occurred while trying to write the vnnlib file {file_path}: {e}")

def write_vnnlib_file(file_path, prop):
    # Derive input and output bounds
    domain = prop["domain"]["bounds"]
    input_bounds = prop_to_input_bounds(domain)

    condition = prop["sat_condition"]
    condition_type = prop["type"]
    output_bounds = prop_to_output_bounds(condition, condition_type)

    # Format bounds into the VNN-LIB format
    vnnlib_content = format_file_content(input_bounds, output_bounds)

    # Write VNN-LIB properties to file
    write_to_file(file_path, vnnlib_content)

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

    # Extract bounds from a specified VNN-LIB file or write one based on the config file
    if "path" in prop:
        properties = parse_vnnlib_file(prop["path"])
    else:
        write_vnnlib_file("./config_property.vnnlib", prop)
        properties = parse_vnnlib_file("./config_property.vnnlib")

    # Change dictionary structure into list of lists
    properties_bounds = []
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

    return properties_bounds
