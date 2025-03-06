import numpy as np

############# Beta Scheduler ################################

def alpha_beta_scheduler(Tmax = 1000,beta_min = 1e-4,beta_max = 3e-3):
    """
    Evaluates all betas/alphas possible and returns as np.array

    ---Input---
    Tmax : (int) Maximum iteration
    beta_min/beta_max : (int) Max and min of beta values

    ---Output---
    betas : (np.array Tmax,) - Scheduled beta
    """

    betas = np.linspace(start = beta_min, stop = beta_max, num = Tmax)
    alphas = 1 - betas
    return alphas,betas

################### Forward DIFFUSION #############################

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
    # Initialize beta
    alphas,betas = alpha_beta_scheduler(Tmax = T,beta_min = beta_min,beta_max = beta_max)
    # Chooses which time point to save for checkpoint
    idx_list = [i for i in range(1,T-1) if i%(T//n_hist) == 0]+[T-1]
    data_hist = []

    n_data,n_dim = data.shape

    for t in range(T):
        if t in idx_list:
            data_hist.append(data)
        
        # Apply variance schedule
        beta = betas[t]
        alpha = alphas[t]

        """
        STAGE 1 :
        # Linear model x_t = alpha_t * x_{t-1} + sigma_t * eps_t
        # alpha_t = sqrt(1-beta), sigma = sqrt(beta)
        # eps_t ~ N(0,I)
        """
        # alpha
        sigma = beta
        # Sample noise
        noise = np.random.multivariate_normal(mean = np.zeros((n_dim,)),
                                              cov = np.identity(n_dim),
                                              size = n_data)
        # One forward operation
        data = alpha * data + sigma * noise
    
    return data_hist,idx_list

def closed_form_forward_target(data, t = 0,beta_min = 1e-4, beta_max = 0.02, Tmax = 1000):
    """
    This function directly transforms the original data x_0 to t-th stage forward transform (close-form)
    linear variance schedule is used

    -Input-
    data (n_dim,) : Single Input data 
    beta_min,beta_max (double) : Bound for beta for variance scheduling
    t (int) : Target iteration
    Tmax (int) : Max iteration

    -Output-
    target_data (n_dim,) : Single output data at epoch t 
    eps (n_dim,) : Noise used to distort (shall be predicted from NN)
    """
    # Initialize beta and alpha
    alphas,betas = alpha_beta_scheduler(Tmax = Tmax, beta_min = beta_min, beta_max = beta_max)

    # Dim
    n_dim = data.shape[0]

    # First extract single beta and get alpha
    beta_t = betas[t]
    alpha_t = alphas[t]

    # Products to get alpha_geom_t
    alpha_geom_t = np.prod(alphas[:t], axis=0)

    # Epsilon
    eps = np.random.normal(size=(n_dim,))
    
    # Evaluate x_t
    x_t = np.sqrt(alpha_geom_t) * data + np.sqrt(1-alpha_geom_t) * eps

    return x_t, eps








