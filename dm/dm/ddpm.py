# ddpm.py
# simple_network.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm.fwd_diffusion import alpha_beta_scheduler
################### ARCHITECTURE ######################################
# Simple Network They say
class SimpleNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, t):
        input_data = torch.hstack([x, t])
        return self.net(input_data)

# Define a simple neural network for DDPM
class DDPM_model(nn.Module):

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

######################## TRAIN, TEST, SAMPLE #########################

def train_ddpm(model,trainloader,optimizer,n_epochs = 10,device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Training DDPM

    ---Input---
    model : (nn.Module) DDPM Model
    trainloader : (torch.util.dataloader) train dataloader
    optimizer : (torch.optim) optimizer
    n_epochs : (int) Number of repeats per batch

    ---Output---
    """
    model.to(device)
    model.train()
    loss_fn = nn.MSELoss()  # Loss function for noise prediction
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        
        for batch in trainloader:
            x, t, eps = batch  # Load batch of data (noisy inputs, timesteps)
            x, t, eps  = x.to(device), t.to(device), eps.to(device) 
            
            optimizer.zero_grad()  # Zero gradients
            
            noise_pred = model(x, t)  # Predict noise
            
            loss = loss_fn(noise_pred, eps)  # Compute loss
            loss.backward()  # Backpropagate
            optimizer.step()  # Update model parameters
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / len(trainloader)}")

def test_ddpm(model,testloader,device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Testing DDPM

    ---Input---
    model : (nn.Module) DDPM Model
    testloader : (torch.util.dataloader) test dataloader

    ---Output---
    """
    # Load model and set to eval mode
    model.to(device)
    model.eval()

    # Initialize loss
    loss_fn = nn.MSELoss()  # Loss function for noise prediction
    total_loss = 0.0

    with torch.no_grad():

      for batch in testloader:
            x, t, eps = batch  # Load batch of data (noisy inputs, timesteps)
            x, t, eps  = x.to(device), t.to(device), eps.to(device) 
            
            noise_pred = model(x, t)  # Predict noise

            loss = loss_fn(noise_pred, eps)  # Compute loss
            total_loss += loss.item()
    
    print(f"Test Loss: {total_loss / len(testloader)}")


def sample_ddpm(model, n_samples = 1000, n_dims = 2, Tmax = 1000, beta_min = 1e-4, beta_max = 3e-3, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Sampling using the DDPM

    ---Input---
    model : (nn.Module) DDPM Model
    n_samples : (int) Number of samples to randomly draw from std normal
    n_dims : (int) Dimension of the data
    Tmax : (int) Maximum steps into forward diffusion 
    beta_min/beta_max : (int) Max and Min Beta to use

    ---Output---
    X_0 : Original sample predicted
    """
    # Save States
    states = [k for k in range(1,Tmax)]
    saved_frame = []
    # Load the model for evaluation
    model.to(device)
    model.eval()

    # Generate the random data X_T
    X_t = torch.randn(size=(n_samples,n_dims)).to(device)

    # Load the betas
    alphas, betas = alpha_beta_scheduler(Tmax = Tmax,beta_min = beta_min,beta_max =beta_max)

    with torch.no_grad():
        # For loop
        for t in reversed(range(1,Tmax)):

            # Time tensor
            t_tensor = torch.full(size=(n_samples,1),fill_value=t/Tmax).to(device)
            
            # alpha and beta for current epoch
            alpha_t,beta_t = alphas[t], betas[t]

            # alpha_bar and beta_bar
            alpha_geom_t_prev = np.prod(alphas[1:t])
            alpha_geom_t = alpha_t * alpha_geom_t_prev

            beta_geom_t = (1-alpha_geom_t_prev)/np.sqrt(1-alpha_geom_t) * beta_t
            
            # model prediction of noise
            X_t = X_t.to(device=device)
            eps_t = model(X_t,t_tensor)

            # Additive noise
            Z_t = torch.randn(size=(n_samples,n_dims)) if t > 1 else 0

            # X_{t-1}
            eps_t = eps_t.to(device="cpu")
            X_t = X_t.to(device="cpu")
            X_t = 1/np.sqrt(alpha_t) * (X_t - (1-alpha_t)/np.sqrt(1-alpha_geom_t) * eps_t) + np.sqrt(beta_t) * Z_t

            if t in states:
              saved_frame.append(X_t.detach().numpy())

    return saved_frame 
