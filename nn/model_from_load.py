import torch
from safetensors.torch import save_model, load_model
import math
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import transformer_block
import load_data

# Model parameters
num_token_options = 3
num_heads = 16
embd_dim = 1024
T = 512 # Sequence length
nlayers = 5

patch_size_x = 16
patch_size_y = 16
im_w = 480
im_h = 320

# Create a new, uninitialized instance of the model
# It's important that this new model has the same architecture as the one we saved
model = transformer_block.TransformerModel(num_token_options, num_heads, embd_dim, T, nlayers)

# Load the saved state_dict into the new model
# The 'strict=True' argument ensures that the keys in the state_dict match the model's keys.
load_model(model, "transformer_model.safetensors", strict=True)
print("Model loaded successfully from safetensors file.")

'''
Training a single step
'''
def calculate_discounted_rewards(rewards, gamma=0.99):
    """Calculates the discounted rewards for an episode."""
    discounted_rewards = []
    cumulative_reward = 0
    for reward in reversed(rewards):
        cumulative_reward = reward + gamma * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)
    # Standardize the rewards (optional but recommended)
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
    return discounted_rewards

states_from_episode = load_data.parse_file_to_tensors("~/step/states.txt")
actions_from_episode = load_data.parse_file_to_list("~/step/actions.txt")
rewards_from_episode = load_data.parse_file_to_list("~/step/rewards.txt")

# --- Optimizer, and Hyperparameters ---
learning_rate = 0.01
gamma = 0.99

# Using the Adam optimizer, maybe muon another day
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- A SINGLE TRAINING STEP ---

# 1. Calculate Discounted Rewards
discounted_rewards = calculate_discounted_rewards(rewards_from_episode, gamma)

# 2. Get Log Probabilities of Actions Taken
p_embd = load_data.create_positional_embeddings_list(im_w/patch_size_x, im_h/patch_size_y, len(states_from_episode))
action_probabilities = model(states_from_episode, im_h, im_w, p_embd)
distribution = Categorical(action_probabilities)
log_probabilities_of_actions = distribution.log_prob(actions_from_episode)

# 3. Calculate the REINFORCE Loss
# The loss is the negative sum of log probabilities multiplied by the rewards
# We want to maximize this objective, so we minimize its negative
loss = -torch.sum(log_probabilities_of_actions * discounted_rewards)

# 4. Use autograd to update the model
optimizer.zero_grad()
loss.backward()
optimizer.step()


print("--- Autograd and Model Update ---")
print("Calculated Loss:", loss.item())
with open('my_file.txt', 'a') as f:
    f.write(f"{len(states_from_episode)}    {loss.item()}\n")

print("\nModel updated successfully using autograd!")

save_model(model, "transformer_model.safetensors")
print("Model saved successfully")