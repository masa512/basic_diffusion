import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def train(model,trainloader,optimizer,n_epochs,device = "cuda" if torch.cuda.is_available() else "cpu"):

    print(f"Using {device} for training... \n")

    # Init protocol
    model.to(device)
    model.train()
    loss_fn = nn.MSELoss()

    # training
    for t in tqdm(range(n_epochs)):

        epoch_loss = 0.0
        
        for batch in trainloader:

            # Load the data to device
            clean_img,noisy_img, label = batch 
            clean_img,noisy_img = clean_img.to(device), noisy_img.to(device)

            # Zero grad first
            optimizer.zero_grad()

            # Model pass and get err
            pred = model(noisy_img)
            loss = loss_fn(pred,clean_img)

            # Grad descent
            loss.backward()  # Backpropagate
            optimizer.step()  # Update model parameters

            # Add to epoch loss
            epoch_loss += loss.item()
        
        # Print current loss
        print(f"Epoch {t}: Avg Loss - {epoch_loss/len(trainloader)}")

    
    


            

          

