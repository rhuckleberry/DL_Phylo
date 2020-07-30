import random

def transform(datapoint):
    """
    Given an datapoint, transforms into all permutations of the datapoint sequences,
    returns 8 alpha, 8 beta, and 8 gamma sequences
    """

    tree_type = datapoint[1]
    all_datapoints = []

    if tree_type == 0:
        #permute alpha sequences
        alpha = datapoint
        alpha_sequences = _alpha_symmetric_permute(alpha)

        all_datapoints += alpha_sequences

        #transform all alpha sequences to gamma and beta
        for alpha_seq in alpha_sequences:
            beta_seq = _alpha_to_beta(alpha_seq)
            gamma_seq = _alpha_to_gamma(alpha_seq)
            all_datapoints.append(beta_seq)
            all_datapoints.append(gamma_seq)

    elif tree_type == 1:
        #permute beta sequences
        beta = datapoint
        #beta_sequences = _beta_permute(beta)
        beta_sequences = [beta]
        all_datapoints += beta_sequences

        #transform all beta sequences to gamma and beta
        # for beta_seq in beta_sequences:
        #     alpha_seq = _beta_to_alpha(beta_seq)
        #     gamma_seq = _beta_to_gamma(beta_seq)
        #     all_datapoints.append(alpha_seq)
        #     all_datapoints.append(gamma_seq)


    elif tree_type == 2:
        #permute gamma sequences
        gamma = datapoint
        #gamma_sequences = _gamma_permute(gamma)
        gamma_sequences = [gamma]
        all_datapoints += gamma_sequences

        #transform all gamma sequences to gamma and beta
        # for gamma_seq in gamma_sequences:
        #     alpha_seq = _gamma_to_alpha(gamma_seq)
        #     beta_seq = _gamma_to_beta(gamma_seq)
        #     all_datapoints.append(alpha_seq)
        #     all_datapoints.append(beta_seq)

    else:
        print("Error: tree_type not correctly defined")



    random.shuffle(all_datapoints)

    return all_datapoints



def _alpha_to_beta(alphaSequence):
    (A, B, C, D), _ = alphaSequence
    return [[A, C, B, D], 1]

def _alpha_to_gamma(alphaSequence):
    (A, B, C, D), _ = alphaSequence
    return [[A, C, D, B], 2]

def _beta_to_alpha(betaSequence):
    (A, B, C, D), _ = betaSequence
    return [[A, C, B, D], 0]

def _beta_to_gamma(betaSequence):
    (A, B, C, D), _ = betaSequence
    return [[A, B, D, C], 2]

def _gamma_to_alpha(gammaSequence):
    (A, B, C, D), _ = gammaSequence
    return [[A, D, B, C], 0]

def _gamma_to_beta(gammaSequence):
    (A, B, C, D), _ = gammaSequence
    return [[A, B, D, C], 1]


def _alpha_permute(alphaSequence):
    """
    Given a datapoint gives all other datapoint sequence alpha_sequencess in the
    same equivalence classes as the input data point
    """
    sequences = alphaSequence[0]
    tree_type = alphaSequence[1]
    assert tree_type == 0

    #flip quartet tree vertically
    (A, B, C, D) = sequences

    return [[[A, B, C, D], tree_type],
            [[A, B, D, C], tree_type],
            [[B, A, C, D], tree_type],
            [[B, A, D, C], tree_type],
            [[C, D, A, B], tree_type],
            [[C, D, B, A], tree_type],
            [[D, C, A, B], tree_type],
            [[D, C, B, A], tree_type]]

def _beta_permute(betaSequence):

    sequences = betaSequence[0]
    tree_type = betaSequence[1]
    assert tree_type == 1

    #flip quartet tree vertically
    (A, B, C, D) = sequences

    return [[[A, B, C, D], tree_type],
            [[C, B, A, D], tree_type],
            [[A, D, C, B], tree_type],
            [[C, D, A, B], tree_type],
            [[B, A, D, C], tree_type],
            [[D, A, B, C], tree_type],
            [[B, C, D, A], tree_type],
            [[D, C, B, A], tree_type]]

def _gamma_permute(gammaSequence):

    sequences = gammaSequence[0]
    tree_type = gammaSequence[1]
    assert tree_type == 2

    #flip quartet tree vertically
    (A, B, C, D) = sequences

    return [[[A, B, C, D], tree_type],
            [[D, B, C, A], tree_type],
            [[A, C, B, D], tree_type],
            [[D, C, B, A], tree_type],
            [[B, A, D, C], tree_type],
            [[C, A, D, B], tree_type],
            [[B, D, A, C], tree_type],
            [[C, D, A, B], tree_type]]
