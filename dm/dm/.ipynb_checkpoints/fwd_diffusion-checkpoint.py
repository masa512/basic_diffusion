import numpy as np
from dm.util import swiss_roll_wrapper

def fwd_diffusion(data,beta_min = 1e-4, beta_max = 0.02, T = 40, n_hist = 1):
    """
    Forward iterative diffusion model
    -Input-
    data (n_data,n_dim) : Input data 
    beta_min,beta_max (double) : Bound for beta for variance scheduling
    T (int) : Stopping iteration
    n_hist (int) : Number of equally spaced history of data evolution excluding last

    -Output-
    #data_mid (n_data,n_dim) : Intermediate value for the data in diffusion
    #data_fin (n_data,n_dim) : Final diffused value
    data_hist (list of data array) : History of the data
    """
    # Chooses which time point to save for checkpoint
    idx_list = [i for i in range(1,T-1) if i%(T//n_hist) == 0]+[T-1]
    data_hist = []

    n_data,n_dim = data.shape

    for t in range(T):
        if t in idx_list:
            data_hist.append(data)
        
        # Apply variance schedule
        beta = beta_min + (beta_max - beta_min) * t/T

        """
        STAGE 1 :
        # Linear model x_t = alpha_t * x_{t-1} + sigma_t * eps_t
        # alpha_t = sqrt(1-beta), sigma = sqrt(beta)
        # eps_t ~ N(0,I)
        """
        # alpha,sigma
        sigma,alpha = beta,np.sqrt(1-beta)
        # Sample noise
        noise = np.random.multivariate_normal(mean = np.zeros((n_dim,)),
                                              cov = np.identity(n_dim),
                                              size = n_data)
        # One forward operation
        data = alpha * data + sigma * noise
    
    return data_hist,idx_list




