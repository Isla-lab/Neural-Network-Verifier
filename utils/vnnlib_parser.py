import json
import os
import re
import time


class VNNLIBParserException(Exception):
    """
    Exception raised for errors while parsing VNN-LIB files

    Attributes:
        message -- explanation of the format error
    """
    def __init__(self, message="Badly formatted VNNLIB file"):
        self.message = message
        super().__init__(self.message)

def read_file(file_path):
    try:
        # Open the file in read mode
        file = open(file_path, 'r')
        return file
    except FileNotFoundError:
        raise VNNLIBParserException(f"The file '{file_path}' does not exist")
    except Exception as e:
        raise VNNLIBParserException(f"An error occurred while trying to open the file {file_path}: {e}")

def get_compact_lines(content):
    # Compact expressions spanning multiple lines into a single line based on the parentheses count
    compact_lines = []

    current_line = ""
    open_parentheses_count = 0

    for idx, line in enumerate(content):
        processed_line = preprocess_line(line)
        open_parentheses = processed_line.count("(")
        close_parentheses = processed_line.count(")")

        open_parentheses_count += open_parentheses - close_parentheses
        current_line += processed_line + " "

        if open_parentheses_count < 0:
            raise VNNLIBParserException(f"Unmatched number of parentheses at line {idx}: {line}")
        elif open_parentheses_count == 0 and current_line.strip():
            current_line = preprocess_line(current_line)
            compact_lines.append(current_line)
            current_line = ""

    return compact_lines

def preprocess_line(line):
    # Remove trailing spaces
    line = line.strip()
    # Remove repeated spaces
    line = re.sub(r'\s+', ' ', line)
    # Remove spaces before and after parentheses
    line = re.sub(r'\s*\(\s*', '(', line)
    line = re.sub(r'\s*\)\s*', ')', line)
    # Remove all content contained within a comment
    line = line.split(';', 1)[0]
    return line

def check_faulty_expression(expression_idx, expression, declaration_pattern, bounds_pattern):
    open_parentheses = expression.count("(")
    close_parentheses = expression.count(")")

    # All open parentheses must be closed within the same expression
    if open_parentheses != close_parentheses:
        raise VNNLIBParserException(f"Unmatched number of parentheses at processed line {expression_idx}: {expression}")
    elif expression:
        # All non-empty expressions must be either a declaration or an assertion
        declaration_matches = list(declaration_pattern.finditer(expression))
        expression_matches = list(bounds_pattern.finditer(expression))

        if not (declaration_matches or expression_matches):
            raise VNNLIBParserException(f"Badly formatted processed line {expression_idx}: {expression}")

def sanity_check(expressions, declaration_pattern, bounds_pattern):
    # Ensure that all expressions are valid or throw an exception
    for idx, expression in enumerate(expressions):
        check_faulty_expression(idx, expression, declaration_pattern, bounds_pattern)

def extract_variable_declarations(expressions, declaration_pattern):
    # Extract the declared variables from the expressions
    variable_declarations = {}
    input_vars = 0
    output_vars = 0

    for expr in expressions:
        match = declaration_pattern.search(expr)
        if match:
            variable, var_type = match.groups()
            if variable not in variable_declarations:
                variable_declarations[variable] = var_type

                if variable.startswith("X_"):
                    input_vars += 1
                else:
                    output_vars += 1

    return variable_declarations, input_vars, output_vars

def sort_dict_by_numeric_keys(input_dict):
    # Extract keys and sort them based on the numeric part
    sorted_keys = sorted(input_dict.keys(), key=lambda k: int(k.split('_')[1]))
    # Create a new dictionary with sorted keys
    sorted_dict = {key: input_dict[key] for key in sorted_keys}
    return sorted_dict

def extract_from_disjunction(constraints, declared_variables):
    new_bounds = []

    # Find all constraints for the current disjunction
    for constraint in constraints:
        bounds = {"inputs": {}, "outputs": {}}

        # Find all constraints for the current junction
        for operator, variable, value in constraint:
            if variable in declared_variables:
                try:
                    converted_value = float(value)
                except ValueError:
                    converted_value = value  # It's a variable name

                # Set upper or lower bound
                if operator == "<=":
                    bound_idx = 1
                else:
                    bound_idx = 0

                # Distinguish input from output constraints
                if variable.startswith('X_'):
                    if variable not in bounds["inputs"]:
                        bounds["inputs"][variable] = [float('-inf'), float('inf')]

                    bounds["inputs"][variable][bound_idx] = converted_value
                else:
                    if variable not in bounds["outputs"]:
                        bounds["outputs"][variable] = [float('-inf'), float('inf')]

                    # Append new acceptable constraints for each output within the same or
                    bounds["outputs"][variable][bound_idx] = converted_value
            else:
                raise VNNLIBParserException(f"Assertion on undeclared variable {variable} in expression: {constraint}")

        new_bounds.append(bounds)

    return new_bounds

def merge_properties(properties, new_bounds):
    # Merge properties with equal input bounds
    merged_properties = {}

    for bound in new_bounds:
        sorted_inputs = sort_dict_by_numeric_keys(bound["inputs"])
        bound_key = json.dumps(sorted_inputs)

        # Join under disjunction all output bounds with equal input bounds
        if len(bound["inputs"]) == 0 and len(bound["outputs"]) > 0:
            for prop in properties:
                prop["outputs"].append(bound["outputs"])
        elif len(bound["outputs"]) > 0:
            if bound_key not in merged_properties:
                merged_properties[bound_key] = [bound["outputs"]]
            else:
                merged_properties[bound_key].append(bound["outputs"])
        else:
            merged_properties[bound_key] = []

    return merged_properties

def propagate_old_properties(properties, new_properties):
    # Join the new properties from the current expression with all previous properties
    for new_prop in new_properties:
        for prop in properties:
            for variable in prop["inputs"].keys():
                if variable not in new_prop["inputs"]:
                    new_prop["inputs"][variable] = [float("-inf"), float("inf")]

                if prop["inputs"][variable][0] > new_prop["inputs"][variable][0]:
                    new_prop["inputs"][variable][0] = prop["inputs"][variable][0]
                if prop["inputs"][variable][1] < new_prop["inputs"][variable][1]:
                    new_prop["inputs"][variable][1] = prop["inputs"][variable][1]
            for output_bounds in prop["outputs"]:
                for new_output_bounds in new_prop["outputs"]:
                    for variable in output_bounds.keys():
                        if variable not in new_output_bounds:
                            new_output_bounds[variable] = output_bounds[variable]

def extract_properties(expressions, declared_variables, bounds_pattern):
    # Regex to capture individual junction constraints
    details_pattern = re.compile(r'\((<=|>=) (X_\d+|Y_\d+) ([^\)]+)\)')

    properties = [{"inputs": {}, "outputs": []}]

    # Find all constraints for all expressions
    for expression in expressions:
        if not bounds_pattern.match(expression):
            continue

        constraints = []

        split_and = re.split(r'\(and', expression)
        for part in split_and:
            constraint = details_pattern.findall(part)
            if constraint:
                constraints.append(constraint)

        new_bounds = extract_from_disjunction(constraints, declared_variables)
        merged_properties = merge_properties(properties, new_bounds)

        # Derive new properties after merging
        new_properties = [{"inputs": json.loads(k), "outputs": v} for k, v in merged_properties.items()]

        propagate_old_properties(properties, new_properties)

        if new_properties:
            properties = new_properties

    for prop in properties:
        prop["inputs"] = sort_dict_by_numeric_keys(prop["inputs"])

    return properties

def add_missing_output_bounds(properties, output_vars):
    # Add inf bounds for unbounded output variables
    for prop in properties:
        for output_bound in prop["outputs"]:
            for var_idx in range(output_vars):
                if f"Y_{var_idx}" not in output_bound:
                    output_bound[f"Y_{var_idx}"] = [float("-inf"), float("inf")]

    for prop in properties:
        for bound_idx in range(len(prop["outputs"])):
            prop["outputs"][bound_idx] = sort_dict_by_numeric_keys(prop["outputs"][bound_idx])

    return properties

def parse_vnnlib_files(path):
    """
    Method that opens the provided VNN-LIB file and parses it to extract properties (input and output bounds)

    Parameters
    ----------
    	path : str
    		The path to the VNN-LIB file to be parsed

    Compute
    ----------
        properties_set : list[list[dict]]
        	A list of lists, each containing a dictionary for a distinct property
        	Each dictionary contains the entries "inputs" and "outputs" for the input and output node boundaries
    Raises
    ------
        VNNLIBParserException
            If an error is encountered while parsing the file or the file is badly formatted
    """
    try:
        if os.path.isdir(path):
            vnn_content_list = []
            for filename in os.listdir(path):
                if filename.endswith('.vnnlib'):
                    file_path = os.path.join(path, filename)
                    vnn_content_list.append(read_file(file_path))
        else:
            vnn_content_list = [read_file(path)]
    except VNNLIBParserException as e:
        print(e)
        return None

    if not vnn_content_list:
        return None

    properties_set = []

    for vnn_content in vnn_content_list:
        # Compactify each expression into a single line
        lines = get_compact_lines(vnn_content)

        # Regex to extract declared variables and asserted input and output boundaries
        declaration_pattern = re.compile(r'\(declare-const (\w+) (\w+)\)')
        bounds_pattern = re.compile(r'\(assert(?:\(or)?(?:\(and)?((?:\(<=|\(>=) (?:X_\d+|Y_\d+) [^\)]+\))*\)?\)?\)')

        # Validate all expressions
        sanity_check(lines, declaration_pattern, bounds_pattern)
        # Extract all declared variables
        declared_variables, input_vars, output_vars = extract_variable_declarations(lines, declaration_pattern)
        # Extract all properties from assertions
        properties = extract_properties(lines, declared_variables, bounds_pattern)
        # Add declared but unbounded output variables
        properties = add_missing_output_bounds(properties, output_vars)

        properties_set.append(properties)

    return properties_set
