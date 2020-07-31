#functions that generate hyperparameters for evolution models:
# - Seq-gen Model: -m
# - Equilibrium Base Frequencies: -f
# - Reversable Rate Matrix: -r
# - Transition Transversion Ratio: -t

import random
import numpy as np

def random_base_frequencies(alpha = (1, 1, 1, 1)):
    """
    Return random sample for dirichlet distribution to get random equilibrium
    base frequencies

    Input:
    alpha - shape of dirichlet distribution

    Output: probability values for base frequencies sampled from dirichlet distribution
    """

    base_freq = ""
    sample = np.random.dirichlet(alpha) #can add size parameter to get more samples

    #return value as a string
    for index, prob_val in enumerate(sample, 1):
        if index == len(sample):
            base_freq += str(prob_val)
        else:
            base_freq += str(prob_val) + ","

    return base_freq

def transition_transversion_ratio(mu = 1, sigma = 2):
    """
    Returns random value for transition_transversion_ratio
    ~Samples from a gaussian distribution and then takes the absolute value
        ~makes smaller values more likely than really large one
    """

    return str(abs(random.gauss(mu, sigma)))

def random_rate_mx(num_rate_mx_parameters, mu = 1, sigma = 0.5, min = 0.001):
    """
    Chooses final index to have value 1. Then chooses random other
    values for the other parameters by a gaussian distribution

    Input: integer representing the number of hyperparameters to randomly assign
    Output: list of values for those hyperparameter rate matrix values
    """
    parameters = []
    #choose values for all indexes except the last one
    for _ in range(num_rate_mx_parameters - 1):
        rate = max(random.gauss(mu, sigma), min)
        parameters.append(rate)

    parameters.append(1) #1 for final index
    assert len(parameters) == num_rate_mx_parameters

    return parameters

def JC():
    """
    *Jukes Cantor
    ~Subcase of HKY
    Returns hyperparameters for jukes_cantor model of evolution
    """
    model = "HKY"
    base_freq = "0.25, 0.25, 0.25, 0.25"
    t_ratio = "0.5"
    rate_mx = None

    return model, base_freq, t_ratio, rate_mx

def K80():
    """
    *Kimura 1980
    ~Subcase of HKY
    Returns hyperparameters for Kimura 1980 model of evolution
    """
    model = "HKY"
    base_freq = "0.25, 0.25, 0.25, 0.25"
    t_ratio = transition_transversion_ratio()
    rate_mx = None

    return model, base_freq, t_ratio, rate_mx

def F81():
    """
    *Felsenstein 1981
    ~Subcase of HKY
    Returns hyperparameters for Felsenstein 1981 model of evolution
    """
    model = "HKY"
    base_freq = random_base_frequencies()
    t_ratio = "0.5"
    rate_mx = None

    return model, base_freq, t_ratio, rate_mx

def HKY85():
    """
    *Hasegawa et al. 1956
    Returns hyperparameters for Hasegawa et al. 1956 model of evolution
    """
    model = "HKY"
    base_freq = random_base_frequencies()
    t_ratio = transition_transversion_ratio()
    rate_mx = None

    return model, base_freq, t_ratio, rate_mx

def F84():
    """
    *Felsenstein 1984
    Returns hyperparameters for Felsenstein 1984 model of evolution
    """
    model = "F84"
    base_freq = random_base_frequencies()
    t_ratio = transition_transversion_ratio()
    rate_mx = None

    return model, base_freq, t_ratio, rate_mx


def GTR():
    """
    Returns hyperparameters for GTR model of evolution
    """
    model = "GTR"
    base_freq = random_base_frequencies()
    t_ratio = None
    pi_1, pi_2, pi_3, pi_4, pi_5, pi_6 = random_rate_mx(6)
    rate_mx = f"{pi_1}, {pi_2}, {pi_3}, {pi_4}, {pi_5}, {pi_6}"

    return model, base_freq, t_ratio, rate_mx
