from sklearn.datasets import make_swiss_roll
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import numpy as np
from dm.fwd_diffusion import alpha_beta_scheduler_torch,closed_form_forward_target_torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


############ DATASET ###############
def swiss_roll_wrapper(n_samples = 100, 
                       sigma_n = 0.0,
                       random_state = None,
                       dim = 2):
    
    """
    Wrapper for generating swiss roll
    Input
    n_samples (int) : # of data
    sigma_n (double) : Noise std dev
    random_state (int) : seed
    dim (int) : Either 3 or 2D

    Returns
    data (n_data, 3 or 2) : Swiss Data
    """

    # Check for dimension
    assert(dim in (2,3)), "Wrong dimension inquiry"

    # use the sklearn function
    x = make_swiss_roll(n_samples=n_samples, noise=sigma_n, random_state=random_state)[0][:,[0,2]]

    # Normalize value
    data = (x - x.mean()) / x.std()
    # check dimension to return
    return data if dim == 3 else data

############# DATASET OBJ ###################

class diffusion_dataset(Dataset):
  def __init__(self, data, Tmax = 1000,beta_min = 1e-4,beta_max=0.02,transforms = None):
    """
    ---Inputs---
    data (array (n,ndim)) : The input data before noising
    Tmax (int) : Maximum iteration
    beta_min,beta_max (double) : Minimum/Maximum beta for scaling
    transforms (list of callables) : List of transformations (None default)

    ---Outputs---
    data_tuple (tuple) : 
      - 0: x_t (torch.tensor(n,ndim)) - dirty output
      - 1: time (int) - random time index
      - 2: eps (torch.tensor(n,ndim)) - the same noise used to distort
    """
    self.data = data
    self.Tmax = Tmax
    self.beta_min = beta_min
    self.beta_max = beta_max
    self.transforms = [torch.from_numpy]

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):

    # Get current data
    x = self.data[idx,:]

    # Apply transformation if provided
    if self.transforms:
      for trans in self.transforms:
        x = trans(x)
    # Extract random index and scale by self.Tmax
    t = torch.randint(0, self.Tmax,size=(1,)).item()

    # Apply the closed form diffusion
    x_t,eps = closed_form_forward_target_torch(x,t,self.beta_min,self.beta_max,self.Tmax)
    
    # Convert all to necessary format and return as tuple
    t_tensor = torch.tensor([t]).to(torch.float32)/(self.Tmax-1)
    x_t_tensor = x_t.to(torch.float32)
    eps_tensor = eps.to(torch.float32)

    output_sample = (x_t_tensor,t_tensor,eps_tensor)
    return output_sample


############## MNIST CUSTOM DATASET #####################

class NoisyMNIST(MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, noise_std=0.2, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.noise_std = noise_std  # Standard deviation of Gaussian noise

    def __getitem__(self, index):
        img, target = super().__getitem__(index)  # Get clean image and label

        # Generate Gaussian noise
        noise = torch.randn_like(img) * self.noise_std
        noisy_img = img + noise  # Add noise to image

        return img, noisy_img, target  # Return clean image, noisy image, and label



############## DATASET/DATALOADER METHODS ####################

def data_split(dataset,train_rate,shuffle=True):
  """
  Splits dataset into training/test set
  
  ---Input---
  dataset (torch.data.dataset) : Dataset created
  train_rate (double) : train proportion (0,1)
  shiffle (bool) : Whether to shuffle data or not

  ---Output---
  train_set (torch.data.dataset)
  test_set (torch.data.dataset)
  """

  train_set, test_set = torch.utils.data.random_split(dataset, [train_rate, 1.0-train_rate])
  return train_set, test_set

def dataloader_wrapper(dataset,batch_size=1,shuffle=True):
  """
  Wrapper for dataloader

  ---Input---
  dataset (torch.data.dataset) : Dataset created
  batch_size (int) : Batch size
  shiffle (bool) : Whether to shuffle data or not

  ---Output---
  dataloader (torch.data.DataLoader) : Output dataloader
  """

  dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
  return dataloader


################# PLOT RELATED #################################

def create_scatter_animation(data_seq,save_path = ".", title = " " ,reverse = False):
    """
    Creates and saves animation from plots

    ---Input---
    data_seq : list of np.array(n_samples,n_dim) for scatter data
    save_path : (str) folder where to save the animation
    title : (str) Title to the plot
    reverse : (bool) whether to traverse backwards

    """

    # Parameters
    n_frames = len(data_seq)

    # Define the f,ax and blank scatter plot obj
    f,ax = plt.subplots(figsize = (5,5))
    scatter = ax.scatter([],[], alpha = 0.1, s = 1)

    # Define init callable (base case when animation begins)
    def init():
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        return (scatter,)
    
    # Define update callable for iterative animation building
    def update(step):
        # Stepping based on whether to reverse traversal or not
        i = step
        if reverse:
          i = (n_frames - 1) - step

        # Update the new frame
        scatter.set_offsets(data_seq[i])
        ax.set_title(f"{title} - {i+1}/{n_frames}")
        return (scatter,)

    # Create animation object
    animation = FuncAnimation(f, update, frames= n_frames, init_func=init, blit=True)

    # Export animation and close the figure
    animation.save(save_path, writer="Justin", fps=10)
    plt.close(f)
    print("Finish saving gif file: ", save_path)

def create_image_animation(img_seq,save_path = ".", title = " " ,reverse = False):
    """
    Creates and saves animation from plots

    ---Input---
    img_seq : list of np.array(dx,dy) for image data
    save_path : (str) folder where to save the animation
    title : (str) Title to the plot
    reverse : (bool) whether to traverse backwards

    """

    # Parameters
    n_frames = len(img_seq)

    # Define the f,ax and blank Image plot obj
    f,ax = plt.subplots(figsize = (5,5))
    im = ax.imshow(img_seq[0], cmap='gray')  # Invisible blank image  

    # Define init callable (base case when animation begins)
    def init():
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title(title)
        return (im,)
    
    # Define update callable for iterative animation building
    def update(step):
        # Stepping based on whether to reverse traversal or not
        i = step
        if reverse:
          i = (n_frames - 1) - step

        # Update the new frame
        im.set_array(img_seq[i])
        ax.set_title(f"{title} - {i+1}/{n_frames}")
        return (im,)

    # Create animation object
    animation = FuncAnimation(f, update, frames= n_frames, blit=True)

    # Export animation and close the figure
    animation.save(save_path, writer="Pillow", fps=10)
    plt.close(f)
    print("Finish saving gif file: ", save_path)
