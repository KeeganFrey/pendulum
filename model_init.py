import torch
from safetensors.torch import save_model
import math
from torch import nn
import torch.nn.functional as F
import transformer_block_FP16
# Model parameters
num_token_options = 3
num_heads = 8
embd_dim = 16
tokens_per_image = 8 * 20
images_per_sequence = 52
T = tokens_per_image * images_per_sequence + images_per_sequence # Sequence length
nlayers = 5

# Create an instance of the model
model_to_save = transformer_block_FP16.TransformerModel(num_token_options, num_heads, embd_dim, T, nlayers)
model_to_save.half()
save_model(model_to_save, "transformer_model_16.safetensors")
