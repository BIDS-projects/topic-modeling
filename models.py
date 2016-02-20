from math import exp, factorial, gamma

def chi_square_model(deg_sep, ALPHA=1.0):
    """
    Applies a chi-square model with ALPHA degrees of freedom.

    WEIGHT = Chi-Square(ALPHA)(DEG_SEP)
    ALPHA: Degrees of freedom of the chi-square distribution.
    ALPHA should satisfy 0 <= ALPHA <= 2 (for monotonocity).
    """
    k = ALPHA / 2.0
    return (deg_sep ** (k - 1)) * exp(- deg_sep / 2.0) / ((2 ** k) * gamma(k))

def exp_decay(deg_sep, ALPHA=2.0):
    """
    Applies an exponential decay model.

    WEIGHT = EXP(- DEG_SEP / ALPHA)
    ALPHA: Parameter to control exponential decay rate.
    """
    return exp(- deg_sep / ALPHA)

def gamma_law(deg_sep, ALPHA=0.5):
    """
    Applies a model using the gamma distribution.

    WEIGHT = Gamma(ALPHA, 1)(DEG_SEP)
    ALPHA: Parameter that controls the shape of the gamma distribution.
    ALPHA should always satisfy 0 <= ALPHA <= 1.
    """
    return (deg_sep ** (ALPHA - 1)) * exp(- deg_sep) / gamma(ALPHA)

def poisson_law(deg_sep, ALPHA=1.0):
    """
    Applies a Poisson distribution.

    WEIGHT = Pois(ALPHA)(DEG_SEP)
    ALPHA = The mean of the Poisson distribution.
    ALPHA must satisfy 0 <= ALPHA <= 1 to satisfy monotonicity.
    """
    return exp(- ALPHA) * (ALPHA ** deg_sep) / factorial(deg_sep)

def power_law(deg_sep, ALPHA=1.0):
    """
    Applies a power law model.

    WEIGHT = DEG_SEP^(-ALPHA)
    ALPHA: Parameter to control power law decay rate. 
    """
    return deg_sep ** (- ALPHA)
