import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


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

    
def test_example(model,dataset,device = "cuda" if torch.cuda.is_available else "cpu"):

    # Load model to device
    model.to(device)
    model.eval()

    with torch.no_grad():

        # Load random index
        i = np.random.randint(len(dataset))

        # Load the data
        clean_img, noisy_img, label = dataset[i]
        clean_img, noisy_img = clean_img.to(device), noisy_img.to(device)

        # Estimate denoised image
        x = noisy_img.unsqueeze(0)
        pred = model(x).detach()

        # Plot all three
        f,ax = plt.subplots(1,3,figsize=(15,5))

        ax[0].imshow(torch.squeeze(clean_img).cpu())
        ax[0].axis('off')

        ax[1].imshow(torch.squeeze(noisy_img).cpu())
        ax[1].axis('off')

        ax[2].imshow(torch.squeeze(pred).cpu())
        ax[2].axis('off')

            

          

