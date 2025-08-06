import torch
import os

#This is used for the actions and the rewards
def parse_file_to_list(file_path):
    """
    Parses a text file where each line contains space-separated integers,
    and converts each line into a 1D PyTorch tensor.

    Args:
        file_path (str): The path to the text file.

    Returns:
        list: A list of 1D PyTorch tensors.
    """
    list_of_numbers = []
    try:
        # Open the file in read mode, the 'with' statement ensures the file is properly closed.
        with open(file_path, 'r') as f:
            for line in f:
                # Remove leading/trailing whitespace and check if the line is not empty.
                stripped_line = line.strip()
                if stripped_line:
                    num = float(stripped_line)
                    list_of_numbers.append(num)

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except ValueError:
        print(f"Error: A line in the file contains non-integer values.")

    return list_of_numbers

#This is used for the images
def parse_file_to_tensors(file_path):
    """
    Parses a text file where each line contains space-separated integers,
    and converts each line into a 1D PyTorch tensor.

    Args:
        file_path (str): The path to the text file.

    Returns:
        list: A list of 1D PyTorch tensors.
    """
    list_of_tensors = []
    try:
        # Open the file in read mode. [1, 2, 3]
        # The 'with' statement ensures the file is properly closed.
        with open(file_path, 'r') as f:
            # Iterate over each line in the file. [5]
            for line in f:
                # Remove leading/trailing whitespace and check if the line is not empty.
                stripped_line = line.strip()
                if stripped_line:
                    # Split the line by spaces to get a list of strings. [4, 8]
                    # Convert each string in the list to an integer. [6, 13]
                    numbers = [int(num) for num in stripped_line.split(' ')]

                    # Convert the list of integers into a 1D PyTorch tensor. [16, 17]
                    tensor = torch.tensor(numbers)

                    # Add the new tensor to our list.
                    list_of_tensors.append(tensor)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except ValueError:
        print(f"Error: A line in the file contains non-integer values.")

    return list_of_tensors

#This assumes that every image is the same size
def create_positional_embeddings_list(num_x, num_y, max_t):
    list_of_tensors = []
    for t in range(max_t):
        initial_tensor = torch.empty(0, 3)
        for y in range(num_y):
            for x in range(num_x):
                tensor = torch.tensor([x,y,t])
                initial_tensor = torch.cat((initial_tensor, tensor.unsqueeze(0)), dim=0)
        list_of_tensors.append(initial_tensor)
    return list_of_tensors