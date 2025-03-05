# ddpm.py
# simple_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

################### ARCHITECTURE ######################################

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
            x, t = batch  # Load batch of data (noisy inputs, timesteps)
            x, t = x.to(device), t.to(device)
            
            optimizer.zero_grad()  # Zero gradients
            
            noise_pred = model(x, t)  # Predict noise
            target_noise = torch.randn_like(x).to(device)  # Ground truth noise (random normal noise)
            
            loss = loss_fn(noise_pred, target_noise)  # Compute loss
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
            x, t = batch  # Load batch of data (noisy inputs, timesteps)
            x, t = x.to(device), t.to(device)
            
            noise_pred = model(x, t)  # Predict noise
            target_noise = torch.randn_like(x).to(device)  # Ground truth noise (random normal noise)
            
            loss = loss_fn(noise_pred, target_noise)  # Compute loss
            total_loss += loss.item()
    
    print(f"Test Loss: {total_loss / len(trainloader)}")





    

  

    return None
