import torch
from safetensors.torch import save_model, load_model
import math
from torch import nn
import torch.nn.functional as F
import transformer_block
# Model parameters
num_token_options = 1000
num_heads = 8
embd_dim = 1024
T = 512 # Sequence length
nlayers = 6

# Create an instance of the model
model_to_save = transformer_block.TransformerModel(num_token_options, num_heads, embd_dim, T, nlayers)
save_model(model_to_save, "transformer_model.safetensors")
