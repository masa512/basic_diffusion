# ddpm.py
from dm.simple_network import simple_DDPM

def ddpm(model,trainloader,optimizer,n_epochs = 10):
  """
  Forward Operation
  """
