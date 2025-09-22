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

    return torch.tensor(list_of_numbers, dtype=torch.float16)

#This is used for the images
def parse_char_file_to_tensors(file_path):
    """
    Parses a text file where each line contains a sequence of characters,
    and converts each line into a 1D PyTorch tensor with FP16 precision.

    Args:
        file_path (str): The path to the text file.

    Returns:
        list: A list of 1D PyTorch tensors of dtype torch.float16.
    """
    list_of_tensors = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    numbers = [float(ord(char))/255.0 for char in stripped_line]

                    # MODIFIED: Specify dtype=torch.float16 during tensor creation.
                    tensor = torch.tensor(numbers, dtype=torch.float16)

                    list_of_tensors.append(tensor)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")

    return list_of_tensors

#This assumes that every image is the same size
#naive implementation with the file wasting space
def create_positional_embeddings_list(num_x, num_y, max_t):
    #print(f"{num_x}    {num_y}    {max_t}")
    list_of_tensors = []
    for t in range(max_t):
        initial_tensor = torch.empty(0, 3)
        for y in range(int(num_y)):
            for x in range(int(num_x)):
                tensor = torch.tensor([x,y,t])
                initial_tensor = torch.cat((initial_tensor, tensor.unsqueeze(0)), dim=0)
        list_of_tensors.append(initial_tensor)
    return list_of_tensors
