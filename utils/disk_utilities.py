import numpy as np
import os


def write_array(array, index, folder_path):
    """
    Write a NumPy array to a binary file.

    Parameters:
    - array: NumPy array to be written.
    - file_path: Path to the binary file.

    Returns:
    None
    """
    
    np.save(f"{folder_path}/temp_{index}.npy", array)


def read_array(folder_path, index):
    """
    Read a subarray from a binary file at a given index.

    Parameters:
    - file_path: Path to the binary file.
    - index: Starting index from which to read.
    - batch_size: Number of elements to read.

    Returns:
    - subarray: Subarray read from the file or an empty array if the index is out of bounds.
    """
    # Check if the file exists
    if not os.path.exists(f"{folder_path}/temp_{index}.npy"):
        return None  # Return None if the file doesn't exist

    array = np.load(f"{folder_path}/temp_{index}.npy")
    
    return array