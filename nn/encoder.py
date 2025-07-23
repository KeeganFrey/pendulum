import torch
import torch.nn as nn

# --- 1. Define Hyperparameters (from previous example) ---
batch_size = 4
num_patches = 196
hidden_dim = 512

# --- Assume we have our patch_embeddings from the encoder ---
patch_embeddings = torch.randn(batch_size, num_patches, hidden_dim)
print(f"Shape of patch embeddings: {patch_embeddings.shape}")

# --- 2. Create the Learnable Positional Embeddings ---
# We need a position for each patch PLUS the [class] token, so num_patches + 1
# We create this as a learnable nn.Parameter.
# Shape: (1, num_patches + 1, hidden_dim)
positional_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
print(f"Shape of positional embeddings parameter: {positional_embeddings.shape}")


# --- 3. Prepare the sequence for the Transformer ---
# a) Create the special [class] token. It's also a learnable parameter.
# Its shape is (1, 1, hidden_dim) so it can be broadcast across the batch.
class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

# b) Expand the class token to match the batch size
# Shape becomes (batch_size, 1, hidden_dim)
class_token_expanded = class_token.expand(batch_size, -1, -1) # -1 means not to change that dimension

# c) Prepend the class token to the patch embeddings
# Shape of patch_embeddings was (4, 196, 512)
# Shape of class_token_expanded is (4, 1, 512)
# Resulting shape is (4, 197, 512)
embeddings_with_class_token = torch.cat((class_token_expanded, patch_embeddings), dim=1)
print(f"Shape after prepending class token: {embeddings_with_class_token.shape}")


# --- 4. Add the Positional Embeddings ---
# The positional_embeddings (shape: 1, 197, 512) are automatically
# broadcast and added to each item in the batch.
input_to_transformer = embeddings_with_class_token + positional_embeddings
print(f"Final shape of input to Transformer: {input_to_transformer.shape}")
