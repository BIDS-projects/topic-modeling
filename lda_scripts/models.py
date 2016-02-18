from math import exp, gamma

def chi_square_model(deg_sep):
    """
    Applies a chi-square model with ALPHA degrees of freedom.
    WEIGHT = Chi-Square(ALPHA)(DEG_SEP)
    """

    # Degrees of freedom of the chi-square distribution.
    # ALPHA should satisfy 0 <= ALPHA <= 2 (for monotonocity).
    ALPHA = 1.0
    k = ALPHA / 2.0

    return (deg_sep ** (k - 1)) * exp(- deg_sep / 2.0) / ((2 ** k) * gamma(k))

def exp_decay(deg_sep):
    """
    Applies an exponential decay model.
    WEIGHT = EXP(- DEG_SEP / ALPHA)
    """

    # Parameter to control exponential decay rate.
    ALPHA = 2.0

    return exp(- deg_sep / ALPHA)

def gamma_law(deg_sep):
    """
    Applies a model using the gamma distribution.
    WEIGHT = Gamma(ALPHA, 1)(DEG_SEP)
    """

    # Parameter that controls the shape of the gamma distribution.
    # ALPHA should always satisfy 0 <= ALPHA <= 1.
    ALPHA = 0.5

    return (deg_sep ** (ALPHA - 1)) * exp(- deg_sep) / gamma(ALPHA)

def poisson_law(deg_sep):
    """
    Applies a Poisson distribution.
    WEIGHT = Pois(ALPHA)(DEG_SEP)
    """

    # The mean of the Poisson distribution.
    # ALPHA must satisfy 0 <= ALPHA <= 1 to satisfy monotonicity.
    ALPHA = 1.0

    return exp(- ALPHA) * (ALPHA ** deg_sep) / gamma(deg_sep)

def power_law(deg_sep):
    """
    Applies a power law model:
    WEIGHT = DEG_SEP^(-ALPHA)
    """

    # Parameter to control power law decay rate.
    ALPHA = 1.0

    return deg_sep ** (- ALPHA)
