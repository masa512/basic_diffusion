from sklearn.datasets import make_swiss_roll
import matplotlib
import matplotlib.pyplot as plt

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


