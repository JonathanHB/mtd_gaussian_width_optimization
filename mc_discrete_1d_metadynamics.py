import numpy as np
from scipy.stats import norm

def monte_carlo_discrete_1d_metadynamics(system, sigma, omega, deltaT, kB, T, nrounds):

    #get information from the system
    x, F = system.evaluate()
    npts = system.npts

    #create reference Gaussian which we can index as a lookup table instead of expensively recalculating the exponential
    ref_gaussian = norm.pdf(x, loc=0, scale=sigma)
    # Normalization condition given that the gaussian includes only the positive numbers
    #A*(2*np.sum(ref_gaussian)-ref_gaussian[0]) = 1  
    ref_gaussian /= (2*np.sum(ref_gaussian)-ref_gaussian[0])  # Normalize the reference Gaussian

    #trajectory and metadynamics potential to track the evolution of the system
    bin_trj = np.zeros(nrounds, dtype=int)
    V = np.zeros((nrounds+1, npts))

    #bin indices used below to select a bin andcalculate minimum image distances
    bin_inds = np.arange(npts)
    #round indices, also returned for convenience
    rounds = np.arange(nrounds)

    for t in rounds:

        #calculate probability density
        prob_unnorm = np.exp(-(F+V[t])/(kB*T))
        probability_density = prob_unnorm/np.sum(prob_unnorm)

        #select a bin
        bin = np.random.choice(bin_inds, p=probability_density) #probabilities must sum to 1
        bin_trj[t] = bin

        #calculate minimum image index distances of all other bins from the selected bin
        minimum_image_index_distances_lower = np.abs(bin-bin_inds)
        minimum_image_index_distances_upper = npts-np.abs(bin-bin_inds)
        minimum_image_index_distances = np.min(np.array([minimum_image_index_distances_lower, minimum_image_index_distances_upper]), axis=0)

        #deposit potential
        V[t+1] = V[t] + omega*np.exp(-V[t,bin]/(kB*deltaT))*ref_gaussian[minimum_image_index_distances]


    return rounds, bin_trj, V