#cython: boundscheck=False, wraparound=False, cdivision=True

"""SNaCK embedding: Stochastic Neighbor and Crowd Kernel embedding.

Works by stapling together the t-SNE (t-distributed Stochastic Neighbor Embedding) and t-STE (t-distributed Stochastic Triplet Embedding) loss functions.

Original MATLAB implementation of tSTE and tSNE: (C) Laurens van der Maaten, 2012, Delft University of Technology

Also uses implementation of t_SNE from scikit-learn. (C) Alexander
Fabisch -- <afabisch@informatik.uni-bremen.de>

Curator: Michael Wilber <mjw285@cornell.eu>

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
cdef extern from "lib-bhtsne/tsne.h":
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

def run_tsne_from_laurens(double [:, ::1] X, int no_dims, double perplexity, double theta = 0.01):
    cdef TSNE t
    Ynp = np.zeros((len(X), no_dims), dtype='double')
    cdef double [:, ::1] Y = Ynp
    t.run(&(X[0,0]), X.shape[0], X.shape[1], &(Y[0,0]), no_dims, perplexity, theta)
    return Ynp

cdef class Inexact_BHTSNE(object):
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

def run_tsne(X_np,
             long[:,::1] triplets,
             int no_dims = 2,
             double perplexity = 30.0,
             double theta = 0.01,
             double contrib_cost_triplets = 1.0,
             double contrib_cost_tsne = 1.0,
             each_fun = None,
             alpha = None,
             int max_iters = 1000,
             int momentum_switch_iter = 250,
             early_exaggeration = 12,
             stop_lying_iter = 250,
             double momentum = 0.5,
             double final_momentum = 0.8,
             double learning_rate = 1.0,
             initial_Y = None,
             verbose = True,
             num_threads = None,
):
    """Learn the triplet embedding for the given triplets.

    Returns an array with shape (max(triplets)+1, no_dims). The i-th
    row in this array corresponds to the no_dims-dimensional
    coordinate of the point.

    Parameters:

    triplets: An Nx3 integer array of object indices. Each row is a
              triplet; first column is the 'reference', second column
              is the 'near edge', and third column is the 'far edge'.
              (MUST BE 0-indexed!!)
    distances: A square distance matrix for t-SNE.
    no_dims:  Number of dimensions in final embedding. High-dimensional
              embeddings are much easier to satisfy (lower training
              error), but may capture less information.
    alpha:    Degrees of freedom in student T kernel. Default is no_dims-1.
              Considered "black magic"; roughly, how much of an impact
              badly satisfying points have on the gradient calculation.
    verbose:  Prints log messages every 10 iterations
    initial_X: The initial set of points to use. Normally distributed if unset.
    num_threads: Parallelism.
    each_function: A function that is called for each gradient update

    """
    def logf(s, *args):
        if verbose: print s%args

    alpha = alpha or no_dims - 1

    cdef int N = X_np.shape[0], D = X_np.shape[1]
    cdef int n, m, i, j, k

    dY_tSNE_np = np.zeros((N, no_dims), 'double')
    dY_tSTE_np = np.zeros((N, no_dims), 'double')
    dY_np = np.zeros((N, no_dims), 'double')
    uY_np = np.zeros((N, no_dims), 'double')
    gains_np = np.zeros((N, no_dims), 'double') + 1.0
    cdef double[:, ::1] dY_tSNE = dY_tSNE_np
    cdef double[:, ::1] dY_tSTE = dY_tSTE_np
    cdef double[:, ::1] dY = dY_np
    cdef double[:, ::1] uY = uY_np
    cdef double[:, ::1] gains = gains_np

    # We modify this in-place
    X_np = X_np.copy()
    cdef double[:, ::1] X = X_np
    X_np -= np.mean(X_np, 0)
    X_np /= np.max(np.abs(X_np))

    if N-1 < 3 * perplexity:
        raise ValueError("Perplexity too large for the number of data points!")

    logf("Using no_dims = %d, perplexity = %f, and theta = %f", no_dims, perplexity, theta)

    exact = (theta == 0)

    if exact:
        tsne_evaluator = Exact_BHTSNE()
    else:
        tsne_evaluator = Inexact_BHTSNE()

    logf("Computing input similarities...")
    tsne_evaluator.calculate_perplexity(X, perplexity)

    # Lie about the P-values
    tsne_evaluator.multiply_p(12)

    # Initialize solution (randomly)
    Y_np = initial_Y if initial_Y is not None else np.random.randn(N, no_dims) * 0.0001
    cdef double[:, ::1] Y = Y_np

    logf("Learning embedding...")
    # if exact:
    #     logf("Learning embedding...")
    # else:
    #     logf("Sparsity = %f! Learning embedding...", row_P[N] / (N*N))

    C_tSNE = -1
    C_tSTE = -1
    for iter in xrange(max_iters):
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
                uY[i,j] = momentum * uY[i,j] - eta * gains[i,j] * dY[i,j]
                # with momentum:
                Y[i,j] = Y[i,j] + uY[i,j]
                # turn off momentum for a moment:
                #Y[i,j] = Y[i,j] - dY[i,j]

        Y_np -= np.mean(Y_np, 0)

        # Stop lying about the P-values after a while, and switch momentum
        if iter == stop_lying_iter:
            tsne_evaluator.multiply_p( 1.0/12.0 )

        if iter == momentum_switch_iter:
            momentum = final_momentum

        if iter%50==0:
            logf("Iter %s", iter)
            # logf("Iteration: %s, error: %s",
            #      iter,
            #      tsne_evaluator.error(Y, theta)
            #      )

        if each_fun:
            each_fun(iter, Y_np, momentum, C_tSTE, C_tSNE)

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
    cdef int i,t,j,k
    cdef double[:, ::1] X = npX
    cdef unsigned int no_triplets = len(triplets)
    cdef double P = 0
    cdef double C = 0
    cdef double A_to_B, A_to_C, const
    cdef double[::1] sum_X = np.zeros((N,), dtype='float64')
    cdef double[:,::1] K = np.zeros((N,N), dtype='double')
    cdef double[:,::1] Q = np.zeros((N,N), dtype='double')
    # Don't need to reinitialize K, Q because they're initialized below in the loop.
    assert K.shape[0] == N; assert K.shape[1] == N
    assert Q.shape[0] == N; assert Q.shape[1] == N
    npdC = np.zeros((N, no_dims), 'float64')
    cdef double[:, ::1] dC = npdC
    cdef double[:, :, ::1] dC_part = np.zeros((no_triplets, no_dims, 3), 'float64')

    # We don't perform L2 regularization, unlike original tSTE
    # (wait, we should do this if we use momentum!)

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
        for i in cython.parallel.prange(N):
            for j in xrange(N):
                K[i,j] = sum_X[i] + sum_X[j]
                for k in xrange(no_dims):
                    K[i,j] += -2 * X[i,k]*X[j,k]
                Q[i,j] = (1 + K[i,j] / alpha) ** -1
                K[i,j] = (1 + K[i,j] / alpha) ** ((alpha+1)/-2)
                # Now, K[i,j] = ((sqdist(i,j)/alpha + 1)) ** (-0.5*(alpha+1)),
                # which is exactly the numerator of p_{i,j} in the lower right of
                # t-STE paper page 3.
                # The proof follows because sqdist(a,b) = (a-b)(a-b) = a^2+b^2-2ab

                # (Note however that we're flipping the long and short
                # edge, since this should be unsatisfied)

        # Compute probability (or log-prob) for each triplet Note that
        # each of these probabilities are FLIPPED; ie. this is the
        # probability that the triplet (a,b,c) is VIOLATED
        for t in cython.parallel.prange(no_triplets):
            P = K[triplets_A[t], triplets_C[t]] / (
                K[triplets_A[t],triplets_B[t]] +
                K[triplets_A[t],triplets_C[t]])
            # This is a mirror image of the equation in the
            # lower-right of page 3 of the t-STE paper. Note that this
            # works because K is some reciprocal of the distance, so
            # I'm convinced this is correct.
            C += P
            # The probability that triplet (a,b,c) is UNSATISFIED.
            # (We want to MINIMIZE C)

            for i in xrange(no_dims):
                # For i = each dimension to use:
                # Calculate the gradient of *this triplet* on its points.
                const = (alpha+1) / alpha
                A_to_B = ((1 - P) *
                          Q[triplets_A[t],triplets_B[t]] *
                          (X[triplets_A[t], i] - X[triplets_B[t], i]))
                A_to_C = ((1 - P) *
                          Q[triplets_A[t],triplets_C[t]] *
                          (X[triplets_A[t], i] - X[triplets_C[t], i]))

                # Problem: Since this is a parallel for loop, we can't
                # accumulate everything at once. Race conditions.
                # So I calculate it once here:
                dC_part[t, i, 0] = const * P * (A_to_B - A_to_C)
                dC_part[t, i, 1] = const * P * (-A_to_B)
                dC_part[t, i, 2] = const * P * (A_to_C)

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
