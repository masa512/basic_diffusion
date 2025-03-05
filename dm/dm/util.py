from sklearn.datasets import make_swiss_roll
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from dm.fwd_diffusion import closed_form_forward_target



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
    data = make_swiss_roll(n_samples=n_samples, noise=sigma_n, random_state=random_state)[0]

    # check dimension to return
    return data if dim == 3 else data[:,[0,2]]

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
      - 0: time (int) - random time index
      - 1: x (torch.tensor(n,ndim)) - clean input
      - 2: x_t (torch.tensor(n,ndim)) - dirty output
    """
    self.data = data
    self.Tmax = Tmax
    self.beta_min = beta_min
    self.beta_max = beta_max
    self.transforms = []

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):

    # Get current data
    x = self.data[idx,:]

    # Apply transformation if provided
    if self.transforms:
      for trans in self.transforms:
        sample = trans(sample)

    # Extract random index and scale by self.Tmax
    t = (np.random.randint(1,self.Tmax)-1)
    
    # Apply the closed form diffusion
    x_t = closed_form_forward_target(x,t,self.beta_min,self.beta_max,self.Tmax)
    
    # Convert all to necessary format and return as tuple
    x_0_tensor = torch.tensor(x).to(torch.float32)
    t_tensor = torch.tensor(t).view(1).to(torch.float32)/(self.Tmax-1)
    x_t_tensor = torch.tensor(x_t).to(torch.float32)
    output_sample = (x_0_tensor,t_tensor,x_t_tensor)
    return output_sample


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

############# Beta Scheduler ################################

def beta_scheduler(Tmax = 1000,beta_min = 1e-4,beta_max = 3e-3):
    """
    Evaluates all betas possible and returns as np.array

    ---Input---
    Tmax : (int) Maximum iteration
    beta_min/beta_max : (int) Max and min of beta values

    ---Output---
    betas : (np.array Tmax,) - Scheduled beta
    """

    betas = np.linspace(start = beta_min, end = beta_max, num = Tmax)
    return betas

