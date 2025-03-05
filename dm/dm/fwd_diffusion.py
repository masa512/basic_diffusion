import numpy as np

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

def closed_form_forward_target(data, t = 0,beta_min = 1e-4, beta_max = 0.02, T = 1000):
    """
    This function directly transforms the original data x_0 to t-th stage forward transform (close-form)
    linear variance schedule is used

    -Input-
    data (n_dim,) : Single Input data 
    beta_min,beta_max (double) : Bound for beta for variance scheduling
    t (int) : Target iteration
    T (int) : Max iteration

    -Output-
    target_data (n_dim,) : Single output data at epoch t 
    """

    n_dim = data.shape[0]

    # Evaluate geometric mean alpha upto t
    alpha_geom_t = 1
    for i in range(t):
      # evaluate alpha_t
      beta_t = beta_min + (beta_max - beta_min) * t/T
      alpha_t = 1-beta_t
      # product
      alpha_geom_t *= alpha_t

    # Epsilon
    eps = np.random.normal(size=(n_dim,))
    
    # Evaluate x_t
    x_t = np.sqrt(alpha_geom_t) * data + np.sqrt(1-alpha_geom_t) * eps

    return x_t







