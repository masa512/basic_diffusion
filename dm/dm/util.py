from sklearn.datasets import make_swiss_roll
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from dm.fwd_diffusion import alpha_beta_scheduler_torch,closed_form_forward_target_torch



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
        sample = trans(sample)
        "Transformed to Torch"

    # Extract random index and scale by self.Tmax
    t = torch.randn(size = (1,self.Tmax))

    # Apply the closed form diffusion
    x_t,eps = closed_form_forward_target_torch(x,t,self.beta_min,self.beta_max,self.Tmax)
    
    # Convert all to necessary format and return as tuple
    t_tensor = t.view(1).to(torch.float32)#/(self.Tmax-1)
    x_t_tensor = x_t.to(torch.float32)
    eps_tensor = eps.to(torch.float32)

    output_sample = (x_t_tensor,t_tensor,eps_tensor)
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


