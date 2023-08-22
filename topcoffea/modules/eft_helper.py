"""Class to help manage calculations involving EFT fits
"""

import numpy as np
import numba
from numba.typed import List
import math


def calc_eft_weights(q_coeffs, wc_values):
    """Calculate the weights for a specific set of WC values.

    Args:
        q_coeffs: Array specifying a set of quadric coefficients parameterizing the weights.
                  The last dimension should specify the coefficients, while any earlier dimensions
                  might be for different histogram bins, events, etc.
        wc_values: A 1D array specifying the Wilson coefficients corrersponding to the desired weight.

    Returns:
        An array of the weight values calculated from the quadratic parameterization.
    """

    # Prepend "1" to the start of the WC array to account for constant and linear terms
    wcs = np.hstack((np.ones(1), wc_values))

    # Initialize the array that will return the coefficients.  It
    # should be the same shape as q_coeffs except missing the last
    # dimension.
    out = np.zeros_like(q_coeffs[..., 0])

    # Now loop over the terms and multiply them out
    index = 1  # start at second column, as first is 0s from boost_histogram underflow (real underflow is row 0)
    for i in range(len(wcs)):
        for j in range(i + 1):
            out += q_coeffs[..., index] * wcs[i] * wcs[j]
            index += 1
    return out

@numba.njit
def n_quad_terms(n_wc):
    """Calculates the number of quadratic terms corresponding to a given
    number of Wilson coefficients.
    """
    return int((n_wc+2)*(n_wc+1)/2)

@numba.njit
def n_wc_from_quad(n_quad):
    """Calculates the number of Wilson coefficients corresponding to a
    given number of quadratic terms
    """
    return int((-3+math.sqrt(9-8*(1-n_quad)))/2)

@numba.njit
def i_to_N(i):
    return int((i+3)*(i+2)*(i+1)*i/24)

@numba.njit
def N_to_i(N):
    return math.floor(math.sqrt((2.5+math.sqrt(6.25-(144-6144*N)/64))/2)-1.5)

@numba.njit
def j_to_N(j):
    return int((j+2)*(j+1)*j/6)

@numba.njit
def N_to_j(N):
    return math.floor(np.around((2*math.sqrt(1/3)*math.cosh(math.acosh(9*math.sqrt(3)*N)/3)-1),5))

@numba.njit
def k_to_N(k):
    return int((k+1)*k/2)

@numba.njit
def N_to_k(N):
    return math.floor((-1+math.sqrt(9-8*(1-N)))/2)


@numba.njit
def quadratic_factors_to_term(factors):
    """Given a 2 element array of which factors will be multipled
    together tells you which element (term) in the quadratic coefficent
    array should be used.
    """

    return factors[1] + k_to_N(factors[0])


@numba.njit
def quadratic_term_to_factors(i_term, factors):
    """Given the term in the quadratic coefficient array, tell you which
    factors should be multiplied together.  Note: to keep from
    reallocating the factors array, we ask the caller to send us an
    array we can fill.
    """

    factors[0] = N_to_k(i_term)
    factors[1] = i_term - k_to_N(factors[0])

@numba.njit
def quartic_factors_to_term(factors):
    """Given a 4 element array of which factors will be multipled
    together tells you which element (term) in the quartic coefficent
    array should be used.
    """

    return int(factors[3] + k_to_N(factors[2]) + j_to_N(factors[1]) + i_to_N(factors[0]))


@numba.njit
def quartic_term_to_factors(i_term, factors):
    """Given the term in the quartic coefficient array, tell you which
    factors should be multiplied together.  Note: to keep from
    reallocating the factors array, we ask the caller to send us an
    array we can fill.
    """

    factors[0] = N_to_i(i_term)
    i_term -= i_to_N(factors[0])
    factors[1] = N_to_j(i_term)
    i_term -= j_to_N(factors[1])
    factors[2] = N_to_k(i_term)
    factors[3] = i_term - k_to_N(factors[2])

@numba.njit
def n_quartic_terms(n_wc):
    return int((n_wc+4)*(n_wc+3)*(n_wc+2)*(n_wc+1)/24)


@numba.njit
def calc_w2_coeffs(q_coeffs, dtype=np.float64):
    """Calculate the quartic coefficients for calculating the w**2 value (needed for histogram errors.

    Args:
        q_coeffs: Array specifying a set of quadric coefficients
                    parameterizing the weights.  The last dimension should
                    specify the coefficients, while any earlier dimensions
                    might be for different histogram bins, events, etc.

    Returns: An array with the quartic coefficients organized
        according to unique terms.  In other words, there is only
        one entry for each unique combination of different powers
        of the various Wilson coefficients.

    """

    n_quad = q_coeffs.shape[-1]
    n_wc = n_wc_from_quad(n_quad)
    w2_coeffs = np.zeros((q_coeffs.shape[:-1])+(n_quartic_terms(n_wc),),dtype)

    # Storage for the factors to multiply these coefficients
    factors = np.zeros(4)

    # Loop over the quadratic terms and multiply them together
    for i in range(n_quad):
        for j in range(i+1):
            coeff = q_coeffs[...,i]*q_coeffs[...,j]
            if i != j:
                coeff *=2
            quadratic_term_to_factors(i,factors[0:2])
            quadratic_term_to_factors(j,factors[2:])

            # Now we need these elements reverse sorted.  I'm really
            # trying to be stingy on memory, so let's do this in place
            # and use the negation trick to sort in reverse order.
            factors*=-1
            factors.sort()
            factors*=-1

            # Figure out which term this contributes to in the final answer
            quartic_term = quartic_factors_to_term(factors)
            w2_coeffs[...,quartic_term] += coeff

    return w2_coeffs

@numba.njit
def calc_eft_w2(quartic_coeffs_unique, wc_values):
    """Calculate the w**2 values for a specific set of WC values.

    Args:
        quartic_coeffs_unique: Array specifying a set of quartic coefficients parameterizing the
                    w**2 values.  Coefficients multiplying redundant terms have been summed.
                    The last dimension should specify the coefficients, while any earlier dimensions
                    might be for different histogram bins, events, etc.
        wc_values: A 1D array specifying the Wilson coefficients corrersponding to the desired weight.

    Returns:
        An array of the w**2 values calculated from the quartic parameterization.
    """
    # Prepend "1" to the start of the WC array to account for constant and linear terms
    wcs = np.hstack((np.array([1]),wc_values))

    # Initilize the array that will return the coefficients.  It
    # should be the same shape as quartic_coeffs_unique, except
    # missing the last dimension
    out = np.zeros_like(quartic_coeffs_unique[...,0])

    # Now loop over the terms and multiply them out
    factors = np.zeros(4,np.int32)

    for i in range(quartic_coeffs_unique.shape[-1]):
        quartic_term_to_factors(i,factors)
        out += quartic_coeffs_unique[...,i]*wcs[factors[0]]*wcs[factors[1]]*wcs[factors[2]]*wcs[factors[3]]

    return out

def remap_coeffs(current_list, target_list, coeffs):
    """Remaps the quadratic fit coefficients to the appropriate order desired for filling a HistEFT.

    Args:
        current_list: The list of WC names for this sample
        target_list: The list of WC names needed to fill the HistEFT
        coeffs: The numpy array of coefficients to remap.  Assume that the last index corresponds to the WC names.

    Returns the fit coefficient values in the ordering defined by
    target_list.  The fit coefficients for any WCs in current_list that
    are omitted from target_list are dropped.  The fit coefficients
    corresponding to any WCs in target_list that don't appear in
    current_list are set to zero.
    """

    # Numba doesn't like Python lists,  Need them as numba typed lists.
    cl = List()
    cl.append('SM')
    for c in current_list:
        cl.append(c)

    tl = List()
    tl.append('SM')
    for t in target_list:
        tl.append(t)

    # The actual logic is inside this compiled code
    return _remap_coeffs(cl, tl, coeffs)


@numba.njit
def _remap_coeffs(current_list, target_list, coeffs):
    # Step one: Define the coefficient mapping
    target_indices = np.zeros(len(target_list),np.int8) # Assuming we have a model with fewer than 256 WCs.  No flavor anarchy here!
    for i, wc in enumerate(target_list):
        try:
            target_indices[i] = current_list.index(wc)
        except:
            target_indices[i] = -1

    # Next, loop over the WC pairs from the target_list and figure out
    # which entry (if any) in the coefficient list they correspond to.
    # Any target_indices value that is negative indicates that value
    # should be filled with zero.
    remapped_coeffs = np.zeros(coeffs.shape[0:-1]+(n_quad_terms(len(target_list)-1),))

    ind = 0
    for i in range(len(target_list)):
        mapped_i = target_indices[i]
        for j in range(i+1):
            mapped_j = target_indices[j]
            # We initialized the values to zero at the start, so we
            # can skip the ones with negative indices here.
            if mapped_i >= 0 and mapped_j >= 0:
                k = quadratic_factors_to_term(sorted((mapped_i,mapped_j),reverse=True))
                remapped_coeffs[...,ind] = coeffs[...,k]
            ind +=1

    return remapped_coeffs
