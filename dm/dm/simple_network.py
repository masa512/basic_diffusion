# simple_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm.fwd_diffusion import closed_form_forward_target

# Define a simple neural network for DDPM
class simple_DDPM(nn.Module):

  # Constructor
  def __init__(self, input_dim, hidden_dim=256, time_emb_dim=128):
    super().__init__()

    # Set up the hyperparams
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.time_emb_dim = time_emb_dim

    # Time embedding layer(two-layer FW NN)
    self.t_embedding = nn.Sequential(
        nn.Linear(1, time_emb_dim), # Takes scalar
        nn.ReLU(),
        nn.Linear(time_emb_dim, time_emb_dim),
    )
    # The main model after time embdedding concatenation
    self.model = nn.Sequential(
        nn.Linear(self.time_emb_dim+self.input_dim, self.hidden_dim),
        nn.ReLU(),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(),
        nn.Linear(self.hidden_dim,self.input_dim)
    )
    
  def forward(self,x,t):
    """
    --- INPUT ---
    x (torch.tensor(n_batch,self.input_dim)) : Noisy input
    t (torch.tensor(n_batch,1)) : Time index

    --- RETURNS ---
    n_pred (torch.tensor(n_batch,self.input_dim)) : Predicted noise
    """

    # First the embedding of time
    time_emb = self.t_embedding(t)

    # Concatenate the time embedding with the input data
    x = torch.cat([x, time_emb], dim=-1)

    # Pass the cat data into the main model
    n_pred = self.model(x)

    return n_pred
