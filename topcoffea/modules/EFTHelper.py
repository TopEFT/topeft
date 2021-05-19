"""Class to help manage calculations involving EFT fits
"""

import numpy as np

class EFTHelper:

    def __init__(self, wc_names):
        """Constructor
        
        Primarily constructs a collection of arrays that will be used to do calculations of w or w**2.
        Args:
            wc_names: Array listing the Wilson coefficients being used
        """

        # Produces an array which tells us which elements from an
        # array which is [1]+wc_values should be multiplied together
        # to calculate the weight from the quadratic coefficients.
        self.quadratic_pairs = np.asarray(np.tril_indices(len(wc_names)+1)).transpose()
        # Tells us which quadratic coefficients should be multiplied together to give the quartic for w**2
        self.w2_pairs = np.asarray(np.tril_indices(len(self.quadratic_pairs))).transpose()
        
        # This next bit is a little convoluted.  We want two things:
        #  1) An array that tells us which WC values to multiply by each term in our quartic function.
        #  2) An array that helps us map each term in the result of squaring the quadratic expression
        #     for the weight to the unique term in the quartic function.
        # On our way there, we'll need a few intermediate arrays.
        # Since we won't use them outside this constructor, we'll just
        # store them in local variables.

        # Tells us values from an array which is [1]+wc_values should
        # be multiplied together to caluclate w**2 from the quartic
        # function.  Note, duplicate terms have not been combined
        # here, so this is just an intermediate item.  The sort is to
        # make it so that we can tell that [0,1,1,1] and [1,0,1,1] are
        # really the same thing.
        quartic_factors = np.sort(
            np.hstack((self.quadratic_pairs[self.w2_pairs[:,0]],
                       self.quadratic_pairs[self.w2_pairs[:,1]])),
            axis=1,
        )
        
        # Removes the duplicate terms from above.  This we'll keep
        # since it will tell us what WC values to combine with each
        # term in order actually to calculate w**2.
        self.quartic_unique_factors = np.unique(quartic_factors,axis=0)
        
        # To get the mapping from the raw results of squaring the
        # quadratic to the more compact form with the appropriate
        # terms combined, we'll a clever way to label which unique
        # term each raw term goes with.  We'll start by calculating a
        # unique number for each unique term that's a function of
        # which WC entries get multipled together.  Lacking a better
        # idea, we'll pretend the four indices into the WC array are
        # really four indices in a 4-D array, and then use the "global
        # index" in that array as a number to identify each term.
        # This is a temporary variable, so we're not storing it.
        quartic_index_list = np.ravel_multi_index(
            (self.quartic_unique_factors[:,0],
             self.quartic_unique_factors[:,1],
             self.quartic_unique_factors[:,2],
             self.quartic_unique_factors[:,3],),
            4*(len(wc_names)+1,)
        )

        # We also need the index values that each term in the raw
        # product (before duplicate removals) corresponds to, so let's
        # use the "ravel_multi_index" trick on that too
        quartic_indices = np.ravel_multi_index(
            (quartic_factors[:,0],
             quartic_factors[:,1],
             quartic_factors[:,2],
             quartic_factors[:,3],),
            4*(len(wc_names)+1,)
        )

        # To get an array that tells us which "bin" the coefficient
        # should be summed in, we need to look up each value in
        # "quartic_indices" and see which array element it maps to in
        # quartic_index_list, which is parallel to
        # "quartic_unique_factors" which the array that lets us
        # multiply everything together.  The function
        # np.searchsorted() will let us accomplish that.
        # This gets used later, so it's saved!
        self.quartic_bins = np.searchsorted(quartic_index_list,quartic_indices)

    def calc_eft_weights(self,q_coeffs,wc_values):
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
        wcs = np.hstack((np.array([1]),wc_values))
        # multiply the WC values by the quadratic coefficiencts and sum to get the weights.
        return np.sum(q_coeffs*wcs[self.quadratic_pairs[:,0]]*wcs[self.quadratic_pairs[:,1]],axis=-1)

    def calc_w2_coeffs(self,q_coeffs):
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

        # First we square the quadratic for calculating the weight
        # value.  Note: this results in multiple terms that would be
        # multipled by the same powers of different Wilson
        # coefficients, so after we do this squaring, we'll be
        # collecting those terms to together to shorten the lenght of
        # the array.  Note 2: The factor of two is because we're only
        # doing the "lower triangle" of the full matrix multiplication
        quartic_coeffs = np.where(self.w2_pairs[:,0]==self.w2_pairs[:,1],
                                  q_coeffs[:,self.w2_pairs[:,0]]*q_coeffs[:,self.w2_pairs[:,1]],
                                  2*q_coeffs[:,self.w2_pairs[:,0]]*q_coeffs[:,self.w2_pairs[:,1]]
                                  )

        # Now we need to resolve the double counting.  We're going to
        # use np.bincount to do the heavy lifting, but the problem is
        # that we might have many entries in q_coeffs (different
        # events) and we don't want to just sum across all of those.
        # There is no way to ask bincount to sum things row-by-row in
        # the matrix, so we'll relabel the bins so that each row has
        # unique bin numbers and then trick bincount into doing our
        # work for us.
        n_unique_terms = len(self.quartic_unique_factors)
        n_terms = quartic_coeffs.shape[1]
        n_rows = quartic_coeffs.shape[0]
        row_ind = np.repeat(np.arange(n_rows),n_terms)
        col_ind = np.tile(self.quartic_bins,n_rows)
        bins = np.ravel_multi_index((row_ind,col_ind,),(n_rows,n_unique_terms))
        # With these bins defined, we can sum the independent coefficients in each row with np.bincount
        quartic_coeffs_unique = np.bincount(bins,weights=quartic_coeffs.flatten(),
                                           minlength=n_rows*n_unique_terms).reshape(n_rows,n_unique_terms)
        return quartic_coeffs_unique

    def calc_eft_w2(self,quartic_coeffs_unique,wc_values):
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
        # multiply the WC values by the quadratic coefficiencts and sum to get the weights.
        return np.sum(quartic_coeffs_unique*
                      wcs[self.quartic_unique_factors[:,0]]*
                      wcs[self.quartic_unique_factors[:,1]]*
                      wcs[self.quartic_unique_factors[:,2]]*
                      wcs[self.quartic_unique_factors[:,3]],
                      axis=-1)
        
        
    def get_w_coeffs(self):
        """Return the number of EFT weight coefficients"""
        return self.quadratic_pairs.shape[0]

    def get_w2_coeffs(self):
        """Return the number of EFT w**2 coefficients"""
        return self.quartic_unique_factors.shape[0]
