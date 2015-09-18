#cython: boundscheck=False, wraparound=False, cdivision=True, embedsignature=True

# (C) Michael Wilber, 2013-2015, UCSD and Cornell Tech.
# All rights reserved. Please see the 'LICENSE.txt' file for details.

"""SNaCK embedding: Stochastic Neighbor and Crowd Kernel embedding.

Works by stapling together the t-SNE (t-distributed Stochastic Neighbor Embedding) and t-STE (t-distributed Stochastic Triplet Embedding) loss functions.

Original MATLAB implementation of tSTE and tSNE: (C) Laurens van der Maaten, 2012, Delft University of Technology

This port (C) Michael Wilber, 2013-2015, UCSD and Cornell Tech.
All rights reserved. Please see the 'LICENSE.txt' file in the source distriution for details.

"""

cimport cython
cimport numpy as np
cimport cython.parallel
cimport openmp

import numpy as np
import os
import gc

from libc cimport math
from libc.stdlib cimport malloc, free

# External wrappers for Barnes Hut t-SNE:
cdef extern from "tsne.h":
    cdef cppclass TSNE:
        void run(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta);
        void computeGradient(double* P, int* inp_row_P, int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta);
        void computeExactGradient(double* P, double* Y, int N, int D, double* dC);
        double evaluateError(double* P, double* Y, int N, int D);
        double evaluateError(int* row_P, int* col_P, double* val_P, double* Y, int N, int D, double theta);
        void computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity)
        void computeGaussianPerplexity(double* X, int N, int D, int** _row_P, int** _col_P, double** _val_P, double perplexity, int K);
        void computeGaussianPerplexity(double* X, int N, int D, int** _row_P, int** _col_P, double** _val_P, double perplexity, double threshold);
        void symmetrizeMatrix(int** _row_P, int** _col_P, double** _val_P, int N)
        void zeroMean(double* X, int N, int D)

cdef class Inexact_BHTSNE(object):
    """Call the inexact version of t-SNE, which approximates the gradient
    by using a tree
    """
    cdef TSNE tsne
    cdef int *row_P
    cdef int *col_P
    cdef double *val_P
    cdef int N

    def calculate_perplexity(self, double [:, ::1] X, double perplexity):
        cdef int N = X.shape[0], D = X.shape[1]
        self.N = N
        self.tsne.computeGaussianPerplexity(<double*> &X[0,0],
                                            <int> N,
                                            <int> D,
                                            <int**> &self.row_P,
                                            <int**> &self.col_P,
                                            <double**> &self.val_P,
                                            <double> perplexity,
                                            <int> int(3 * perplexity))

        self.tsne.symmetrizeMatrix(&self.row_P, &self.col_P, &self.val_P, N)

        cdef double sum_P = .0
        for i in xrange(self.row_P[N]):
            sum_P += self.val_P[i]
        for i in xrange(self.row_P[N]):
            self.val_P[i] /= sum_P

    def multiply_p(self, double factor):
        cdef int i
        for i in xrange(self.row_P[self.N]):
            self.val_P[i] *= factor

    def calculate_gradient(self, double [:, ::1] Y, int no_dims, double[:, ::1] dY, double theta):
        self.tsne.computeGradient(NULL, self.row_P, self.col_P, self.val_P, &Y[0,0], Y.shape[0], no_dims, &dY[0,0], theta)

    def error(self, double[:, ::1] Y, theta):
        return self.tsne.evaluateError(<int*>self.row_P,
                                       <int*>self.col_P,
                                       <double*>self.val_P,
                                       <double*>&Y[0,0],
                                       <int>Y.shape[0],
                                       <int>Y.shape[1],
                                       <double>theta)

    def destroy(self):
        free(self.row_P)
        free(self.col_P)
        free(self.val_P)
        self.row_P = NULL
        self.col_P = NULL
        self.val_P = NULL

cdef class Exact_BHTSNE(object):
    """Exact version of Barnes-Hut t-SNE. Warning: takes O(N^2) memory
    and time, where N is the number of points.
    """
    cdef TSNE tsne
    cdef object P_np

    def calculate_perplexity(self, double [:, ::1] X, double perplexity):
        cdef int N = X.shape[0], D = X.shape[1]
        # Compute similarities
        self.P_np = np.zeros((N,N), 'double')
        cdef double[:, ::1] P = self.P_np
        self.tsne.computeGaussianPerplexity(&X[0,0], N, D, &P[0,0], perplexity)
        cdef int n,m
        for n in xrange(N):
            for m in xrange(n+1, N):
                P[n,m] += P[m,n]
                P[m,n] = P[n,m]

        cdef double sum_P = .0
        for i in xrange(N):
            for j in xrange(N):
                sum_P += P[i,j]
        for i in xrange(N):
            for j in xrange(N):
                P[i,j] = P[i,j] / sum_P

    def multiply_p(self, double factor):
        self.P_np *= factor

    def calculate_gradient(self, double [:, ::1] Y, int no_dims, double[:, ::1] dY, double theta):
        cdef double[:, ::1] P = self.P_np
        self.tsne.computeExactGradient(&P[0,0], &Y[0,0], Y.shape[0], no_dims, &dY[0,0])

    def destroy(self):
        pass # the gc will get self.P_np

    def error(self, double[:, ::1] Y, theta):
        cdef double[:,::1] P = self.P_np
        return self.tsne.evaluateError(&P[0,0], &Y[0,0], Y.shape[0], Y.shape[1])

def snack_embed(
        # Important:
        double[:, ::1] X_np,
        double contrib_cost_tsne,
        long[:,::1] triplets,
        double contrib_cost_triplets,
        double perplexity = 30.0,
        double theta = 0,

        # Fine-grained optimization control:
        int no_dims = 2,
        alpha = None,
        int max_iter = 300,     # BH-tSNE default: 1000
        early_exaggeration = 4, # BH-tSNE default: 12
        early_exaggeration_switch_iter = 100,  # BH-tSNE default: 250 (see max_iter!)
        double momentum = 0.5,
        int momentum_switch_iter = 250,
        double final_momentum = 0.8,
        double learning_rate = 1.0,
        initial_Y = None,

        # Other:
        each_fun = None,
        verbose = True,
        num_threads = None,
):
    """Learn the triplet embedding for the given triplets.

    Returns an array with shape (len(X_np), no_dims). The i-th
    row in this array corresponds to the no_dims-dimensional
    coordinate of the point.

    Important parameters
    --------------------

    X_np : numpy array with shape (N, D) and type double
        The feature representation of N points in D dimensions.
    contrib_cost_tsne : double
        Trade-off: raises the influence of t-SNE in the final
        embedding. Typical ranges are between 100 and 5000.
        Suggestion: Pick contrib_cost_tsne and contrib_cost_triplets
        such that the norms of the corresponding gradients are roughly
        equal.
    triplets : numpy array with shape (T, 3) and type long.
        Each row is (a,b,c), which indicates that in the desired
        embedding Y, then object a should be closer to b than a is to
        C, or |Y[a]-Y[b]| < |Y[a]-Y[c]|.
    contrib_cost_triplets: double
        Trade-off: raises the influence of triplets in the final
        embedding. Typical ranges are between 0.1 and 5 Suggestion:
        Pick contrib_cost_tsne and contrib_cost_triplets such that the
        norms of the corresponding gradients are roughly equal.
    perplexity : double
        t-SNE perplexity parameter, controlling the number of expected
        neighbors for each point. Generally between 10 and 300.
    theta : double
        Optimization parameter: set to 0.0 for an exact solution.
        Higher values are faster and sloppier. Sets of points whose
        width is smaller than this angle will be collapsed. Try
        starting with theta=0.5 and adjust as needed.

    Optimization parameters
    -----------------------

    no_dims : int
        Number of dimensions of the final embedding. High-dimensional
        embeddings are much easier to satisfy (lower training error),
        but may capture less information. (UNTESTED for >2 dimensions)
    alpha : int
        Degrees of freedom in student T kernel for t-STE. Default is
        no_dims-1. Considered "black magic"; roughly, how much of an
        impact badly satisfying points have on the gradient
        calculation.
    max_iter : int
        Number of iterations to convergence. 300 or 500 is usually enough.
    early_exaggeration : double
        Magic t-SNE parameter. (see paper)
    early_exaggeration_switch_iter : int
        Which iteration to stop using early exaggeration.
    momentum : double
        Magic optimization.
    final_momentum : double
    momentum_switch_iter : int
        After momentum_switch_iter iterations, switch momentum to the
        value of final_momentum.
    learning_rate : double
        All gradients are multiplied by this value.
    initial_y : numpy array with shape (N, no_dims) and type double.
        Matrix with shape (N, no_dims). If not specified, the
        embedding will be calculated completely randomly. If
        unspecified, embedding will be created randomly.

    Other parameters
    ----------------

    each_fun : function
        Callback called on each iteration. Accepts parameters:
        (iteration number, current_Y, C_tSTE, C_tSNE, dY_tSTE,
        dY_tSNE) Note current_Y is modified each iteration. C_tSTE()
        and C_tSNE() are thunks that calculate tSTE and tSNE error
        when called. (This may be expensive!) dY_tSTE and dY_tSNE are
        the gradients of tSTE and tSNE.
    verbose : boolean
        Whether to log debug information. Note that BH-tSNE also does
        its own logging to stderr from C++.
    num_threads : int
        Number of threads to use for OpenMP (but this doesn't really
        matter much anymore; only the t-STE calculation is
        parallelized)

    """
    def logf(s, *args):
        if verbose: print s%args

    # Set number of threads
    if num_threads is None:
        num_threads = openmp.omp_get_num_procs()
    openmp.omp_set_num_threads(num_threads)

    # Set up variables
    cdef int N = X_np.shape[0], D = X_np.shape[1]
    cdef int n, m, i, j, k
    dY_tSNE_np = np.zeros((N, no_dims), 'double')
    dY_tSTE_np = np.zeros((N, no_dims), 'double')
    dY_np = np.zeros((N, no_dims), 'double')
    uY_np = np.zeros((N, no_dims), 'double')
    gains_np = np.zeros((N, no_dims), 'double') + 1.0
    Y_np = initial_Y if initial_Y is not None else np.random.randn(N, no_dims) * 0.0001
    cdef double[:, ::1] dY_tSNE = dY_tSNE_np # Save gradient wrt. tSNE
    cdef double[:, ::1] dY_tSTE = dY_tSTE_np # Save gradient wrt. tSTE (triplets)
    cdef double[:, ::1] dY = dY_np           # Combined gradient (instantaneous)
    cdef double[:, ::1] uY = uY_np           # Velocity (just the accumulated gradient with momentum)
    cdef double[:, ::1] gains = gains_np     # Momentum hack to quickly stop points going in the wrong direction
    cdef double[:, ::1] Y = Y_np             # FINAL SOLUTION
    alpha = alpha or no_dims - 1             # Degrees of freedom in Student-T kernel
    exact = (theta == 0)
    tsne_evaluator = Exact_BHTSNE() if exact else Inexact_BHTSNE()
    # Set up X (standardization)
    X_np = X_np.copy()
    cdef double[:, ::1] X = X_np
    X_np -= np.mean(X_np, 0)
    X_np /= np.max(np.abs(X_np))
    # Normalize triplet cost by the number of triplets that we have
    contrib_cost_triplets = contrib_cost_triplets*(2.0 / float(len(triplets)) * float(N))

    # Warnings
    assert -1 not in triplets
    assert np.max(triplets) <= N, "Some triplets refer to nonexistent points."
    if N-1 < 3 * perplexity:
        raise ValueError("Perplexity too large for the number of data points!")

    logf("Using no_dims = %d, perplexity = %f, and theta = %f", no_dims, perplexity, theta)
    logf("Computing input similarities...")
    tsne_evaluator.calculate_perplexity(X, perplexity)

    # Lie about the P-values
    # (this will be undone later)
    tsne_evaluator.multiply_p(early_exaggeration)

    logf("Learning embedding...")

    # Gradient descent!!
    C_tSNE = lambda: -1
    C_tSTE = lambda: -1
    for iter in xrange(max_iter):
        if contrib_cost_tsne:
            tsne_evaluator.calculate_gradient(Y, no_dims, dY_tSNE, theta)
            C_tSNE = lambda: tsne_evaluator.error(Y, theta)
            # calculating the error here is surprisingly expensive!!
            # so we wrap it in a thunk, just in case...

        if contrib_cost_triplets:
            C_tSTE_val, dY_tSTE = tste_grad(Y, N, no_dims, triplets, alpha)
            C_tSTE = lambda: C_tSTE_val
            # thunk, for symmetricity with C_tSNE

        # Set gradient
        for i in xrange(N):
            for j in xrange(no_dims):
                dY[i,j] = (contrib_cost_triplets * dY_tSTE[i,j] +
                           contrib_cost_tsne * dY_tSNE[i,j])

        # Update gains
        for i in xrange(N):
            for j in xrange(no_dims):
                gains[i,j] = gains[i,j] + 0.2 if (dY[i,j]*uY[i,j] < 0) else gains[i,j]*0.8
                if gains[i,j] < 0.01:
                    gains[i,j] = 0.01

        # Perform gradient update with momentum and gains
        for i in xrange(N):
            for j in xrange(no_dims):
                uY[i,j] = momentum * uY[i,j] - learning_rate * gains[i,j] * dY[i,j]
                # with momentum:
                Y[i,j] = Y[i,j] + uY[i,j]
                # turn off momentum for a moment:
                #Y[i,j] = Y[i,j] - dY[i,j]

        # Standardize
        Y_np -= np.mean(Y_np, 0)

        # Stop lying about the P-values after a while, and switch momentum
        if iter == early_exaggeration_switch_iter:
            tsne_evaluator.multiply_p( 1.0/early_exaggeration )

        if iter == momentum_switch_iter:
            momentum = final_momentum

        if iter%50==0:
            logf("Iter %s", iter)

        if each_fun:
            each_fun(iter, Y_np, C_tSTE, C_tSNE, dY_tSTE, dY_tSNE)

    tsne_evaluator.destroy()
    return Y_np


cpdef tste_grad(npX,
                int N,
                int no_dims,
                long [:, ::1] triplets,
                double alpha,
):
    """Compute the cost function and gradient update of t-STE.

    Note that this is a bit different from t-STE's paper! Our utility
    function (NOT cost!) is how many "incorrect" triplets are
    satisfied. We then want to minimize the number of unsatisfied
    triplets.

    The gradient seems OK --- see check_gradient.py here.

    Parameters
    ----------
    npX : The embedding so far. Should have shape (N, no_dims)
    N : Number of points to embed, should be equal to len(npX) I think (oops)
    no_dims : Dimensionality of the embedding
    triplets : Crowdsourced triplets, with shape (N, 3). List of
               (a,b,c)-tuples where a is closer to b than a is to c
    alpha : Degrees of freedom in the Student-T kernel.
    """
    cdef long[:] triplets_A = triplets[:,0]
    cdef long[:] triplets_B = triplets[:,1]
    cdef long[:] triplets_C = triplets[:,2]
    cdef int i,t,j,k,d
    cdef double[:, ::1] X = npX
    cdef unsigned int no_triplets = len(triplets)
    cdef double P = 0
    cdef double C = 0
    cdef double A_to_B, A_to_C, const
    cdef double[::1] sum_X = np.zeros((N,), dtype='float64')
    cdef double Qij, Qik, Kij, Kik
    npdC = np.zeros((N, no_dims), 'float64')
    cdef double[:, ::1] dC = npdC
    cdef double[:, :, ::1] dC_part = np.zeros((no_triplets, no_dims, 3), 'float64')

    # We don't perform L2 regularization, unlike original tSTE
    # (...but perhaps we should do this if we use momentum!)

    # # L2 Regularization cost
    # cdef double lamb = 1.0 # (No regularization!)
    # for i in xrange(N):
    #     for j in xrange(no_dims):
    #         C += X[i,j]*X[i,j]
    # C *= lamb

    # Compute student-T kernel for each point
    # i,j range over points; k ranges over dims
    with nogil:
        for i in xrange(N):
            sum_X[i] = 0
            for k in xrange(no_dims):
                # Squared norm
                sum_X[i] += X[i,k]*X[i,k]

        # Compute probability (or log-prob) for each triplet Note that
        # each of these probabilities are FLIPPED; ie. this is the
        # probability that the triplet (a,b,c) is VIOLATED
        for t in cython.parallel.prange(no_triplets):
            i = triplets_A[t]
            j = triplets_B[t]
            k = triplets_C[t]
            # Compute short arm distance
            Kij = sum_X[i] + sum_X[j]
            for d in xrange(no_dims):
                Kij += -2 * X[i,d]*X[j,d]
            Qij = (1 + Kij / alpha) ** -1
            Kij = (1 + Kij / alpha) ** ((alpha+1)/-2)
            # Compute long arm distance
            Kik = sum_X[i] + sum_X[k]
            for d in xrange(no_dims):
                Kik += -2 * X[i,d]*X[k,d]
            Qik = (1 + Kik / alpha) ** -1
            Kik = (1 + Kik / alpha) ** ((alpha+1)/-2)

            #Kik *= 0.5

            # Now, Kij = ((sqdist(i,j)/alpha + 1)) ** (-0.5*(alpha+1)),
            # which is exactly the numerator of p_{i,j} in the lower right of
            # t-STE paper page 3.
            # The proof follows because sqdist(a,b) = (a-b)(a-b) = a^2+b^2-2ab
            # (Note however that we're flipping the long and short
            # edge, since this should be unsatisfied)
            P = Kik / (Kij + Kik)
            # This is a mirror image of the equation in the
            # lower-right of page 3 of the t-STE paper. Note that this
            # works because K is some reciprocal of the distance, so
            # I'm convinced this is correct.
            C += P
            # The probability that triplet (a,b,c) is UNSATISFIED.
            # (We want to MINIMIZE C)

            for d in xrange(no_dims):
                # For d = each dimension to use:
                # Calculate the gradient of *this triplet* on its points.
                const = (alpha+1) / alpha
                A_to_B = ((1 - P) * Qij *
                          (X[triplets_A[t], d] - X[triplets_B[t], d]))
                A_to_C = ((1 - P) * Qik *
                          (X[triplets_A[t], d] - X[triplets_C[t], d]))

                # Problem: Since this is a parallel for loop, we can't
                # accumulate everything at once. Race conditions.
                # So I calculate it once here:
                dC_part[t, d, 0] = const * P * (A_to_B - A_to_C)
                dC_part[t, d, 1] = const * P * (-A_to_B)
                dC_part[t, d, 2] = const * P * (A_to_C)

                # This is like 'use_log=False', which doesn't make
                # sense in this inverted "minimize number of
                # unsatisfied triplets" formulation.

        # ...and then accumulate:
        for n in xrange(N):
            for i in xrange(no_dims):
                dC[n, i] = 0
        for t in xrange(no_triplets):
            for i in xrange(no_dims):
                dC[triplets_A[t], i] += dC_part[t, i, 0]
                dC[triplets_B[t], i] += dC_part[t, i, 1]
                dC[triplets_C[t], i] += dC_part[t, i, 2]
        # # L2 regularization
        # for n in xrange(N):
        #     for i in xrange(no_dims):
        #         # The 2*lamb*npx is for regularization: derivative of L2 norm
        #         dC[n,i] = dC[n,i] + 2*lamb*X[n,i]
    return C, npdC
